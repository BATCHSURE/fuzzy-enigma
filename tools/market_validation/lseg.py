"""Bounded, historical SPXW snapshots through an explicit LSEG Desktop session.

The optional LSEG dependency is imported only when a request is made.  Credentials
come from LSEG_APP_KEY (or doctor's hidden prompt), never a configuration file.
"""

from __future__ import annotations

import csv
import getpass
import hashlib
import importlib
import json
import math
import os
import re
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


_TERMS_SOURCE = "https://www.cboe.com/tradable-products/sp-500/spx-options/spx-specifications"
_SURFACE_URL = "https://api.refinitiv.com/data/quantitative-analytics-curves-and-surfaces/v1/surfaces"
_CONTRACT_URL = "https://api.refinitiv.com/data/quantitative-analytics/v1/financial-contracts"
_CONTRACT_DOC = "https://developers.lseg.com/en/api-catalog/refinitiv-data-platform/refinitiv-data-platform-apis/documentation/manuals-and-guides/ipa-financial-contracts/ipa-financial-contracts-option-contracts-eti"
_SURFACE_DOC = "https://developers.lseg.com/en/api-catalog/refinitiv-data-platform/refinitiv-data-platform-apis/documentation/manuals-and-guides/ipa-volatility-surfaces/ipa-volatility-surfaces-eti"
_REQUIRED_SEARCH = {
    "RIC", "OptionStub", "UnderlyingQuoteRIC", "ExpiryDate", "StrikePrice", "CallPutOption"
}
_PAGE_SIZE = 1000
_SEARCH_LIMIT = 10000


class MarketDataError(RuntimeError):
    """A safe, actionable acquisition failure, with a stable machine-readable code."""

    def __init__(self, code, message, diagnostics=None):
        super().__init__(message)
        self.code = code
        self.diagnostics = diagnostics or {}


def _diagnostic(exc, secret=None):
    message = str(exc)
    secrets = [secret] if secret else []
    secrets.extend(
        value for key, value in os.environ.items()
        if key.startswith("LSEG_") and any(word in key for word in ("KEY", "TOKEN", "SECRET", "PASSWORD"))
    )
    for value in secrets:
        if value:
            message = message.replace(value, "[REDACTED]")
    # Transport exception strings can include authentication query parameters.
    message = re.sub(
        r"(?i)(app[-_]?key|access[-_]?token|client[-_]?secret|password)([\s=:]+)[^\s,;}&]+",
        r"\1\2[REDACTED]", message,
    )
    message = re.sub(r"\b[0-9a-fA-F]{40}\b", "[REDACTED]", message)
    result = {"status": "error", "code": getattr(exc, "code", type(exc).__name__), "message": message[:600]}
    if getattr(exc, "diagnostics", None):
        result["details"] = _safe_raw(exc.diagnostics)
    return result


def _safe_raw(value):
    """Keep response data, excluding transport headers and credential-shaped keys."""
    if isinstance(value, dict):
        return {
            str(key): _safe_raw(item) for key, item in value.items()
            if not any(word in _norm(key) for word in ("token", "password", "secret", "appkey", "authorization"))
            and _norm(key) not in {"requestheaders", "httpheaders", "responseheaders"}
        }
    if isinstance(value, (list, tuple)):
        return [_safe_raw(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str):
        for key, secret in os.environ.items():
            if key.startswith("LSEG_") and any(word in key for word in ("KEY", "TOKEN", "SECRET", "PASSWORD")) and secret:
                value = value.replace(secret, "[REDACTED]")
        return value
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if hasattr(value, "item"):
        return _safe_raw(value.item())
    return str(value)


def _norm(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    try:
        return date.fromisoformat(text[:10])
    except (ValueError, TypeError):
        raise MarketDataError("invalid_date", f"Invalid ISO date: {text[:30]}") from None


def _positive(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise MarketDataError("invalid_number", f"{label} must be a finite positive number") from None
    if not math.isfinite(number) or number <= 0:
        raise MarketDataError("invalid_number", f"{label} must be a finite positive number")
    return number


@contextmanager
def _desktop(*, prompt=False):
    try:
        ld = importlib.import_module("lseg.data")
    except ImportError:
        raise MarketDataError("dependency_missing", "Install the optional market dependencies (lseg-data) first") from None
    key = os.environ.get("LSEG_APP_KEY", "").strip()
    if not key and prompt:
        key = getpass.getpass("LSEG Desktop App Key: ").strip()
    if not key:
        raise MarketDataError("app_key_missing", "Set LSEG_APP_KEY, or use doctor with the hidden prompt")
    session = None
    try:
        session = ld.session.desktop.Definition(app_key=key).get_session()
        session.open()
        yield session, ld, key
    except MarketDataError:
        raise
    except Exception as exc:
        details = _diagnostic(exc, key)
        raise MarketDataError("lseg_request_failed", details["message"], details) from None
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                # A failed close must not erase an already diagnosed acquisition error.
                pass


def _frame(response, label):
    _check_response(response, label)
    data = getattr(response, "data", None)
    frame = getattr(data, "df", None)
    if frame is None or frame.empty:
        raise MarketDataError("no_data", f"{label} returned no usable data; check content permissions and date coverage")
    return frame


def _check_response(response, label):
    if getattr(response, "is_success", True) is False:
        errors = getattr(response, "errors", None)
        detail = _diagnostic(RuntimeError(str(errors))) if errors else {}
        raise MarketDataError("provider_rejected", f"{label} was rejected by LSEG; check permissions, coverage and returned diagnostics", detail)


def _reference(session, universe, fields, parameters):
    from lseg.data.content import fundamental_and_reference

    return fundamental_and_reference.Definition(
        universe=universe, fields=fields, parameters=parameters,
    ).get_data(session=session)


def _column(frame, name, *aliases):
    candidates = {_norm(item) for item in (name,) + aliases}
    for column in frame.columns:
        if _norm(column) in candidates:
            return column
    raise MarketDataError("missing_field", f"Response did not contain required field {name}")


def _underlying_close(session, requested=None):
    # Never request today's incomplete close.  The actual returned close date,
    # rather than a weekday calculation, determines the latest trading date.
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=1)
    target = _date(requested) if requested is not None else cutoff
    if target > cutoff:
        raise MarketDataError("incomplete_date", "as_of must be a completed historical date before today (UTC)")
    start = target if requested is not None else target - timedelta(days=14)
    response = _reference(session, [".SPX"], ["TR.PriceClose", "TR.PriceClose.date"], {
        "SDate": start.isoformat(), "EDate": target.isoformat(), "Frq": "D", "Curn": "USD",
    })
    frame = _frame(response, "SPX historical close")
    prices = _column(frame, "TR.PriceClose", "Price Close", "Close Price")
    dates = _column(frame, "TR.PriceClose.date", "Date", "Price Close Date")
    closes = []
    for _, row in frame.iterrows():
        try:
            observed = _date(row[dates])
            price = _positive(row[prices], "SPX close")
        except MarketDataError:
            continue
        if observed <= target and (requested is None or observed == target):
            closes.append((observed, price))
    if not closes:
        raise MarketDataError("historical_close_missing", "SPX did not return a positive close dated exactly as requested")
    observed, price = max(closes)
    return observed, price, _safe_raw(getattr(response.data, "raw", None))


def _search_metadata(session):
    from lseg.data.content import search

    response = search.metadata.Definition(view=search.Views.EQUITY_DERIVATIVE_QUOTES).get_data(session=session)
    frame = _frame(response, "Search metadata")
    names = {str(value) for value in frame.index if isinstance(value, str)}
    for column in frame.columns:
        if _norm(column) in {"property", "propertyname", "name"}:
            names.update(str(value) for value in frame[column].dropna())
    raw = getattr(response.data, "raw", None)
    if isinstance(raw, dict):
        properties = raw.get("Properties", raw.get("properties", {}))
        if isinstance(properties, dict):
            names.update(properties)
        elif isinstance(properties, list):
            for item in properties:
                if isinstance(item, dict):
                    names.add(str(item.get("Name", item.get("name", ""))))
    missing = sorted(_REQUIRED_SEARCH - names)
    if missing:
        raise MarketDataError("search_metadata_missing", "Search view lacks required SPXW properties: " + ", ".join(missing))
    return names


def _search_pages(session, names, filter_text):
    from lseg.data.content import search

    selected = sorted(_REQUIRED_SEARCH | ({"Currency"} if "Currency" in names else set()))
    rows, seen = [], set()
    for offset in range(0, _SEARCH_LIMIT, _PAGE_SIZE):
        response = search.Definition(
            view=search.Views.EQUITY_DERIVATIVE_QUOTES, filter=filter_text,
            select=",".join(selected), top=_PAGE_SIZE, skip=offset,
            order_by="RIC asc",
        ).get_data(session=session)
        _check_response(response, "SPXW Search")
        data = getattr(response, "data", None)
        frame = getattr(data, "df", None)
        if frame is None:
            raise MarketDataError("search_failed", "Search returned no data object; check Search permission")
        if frame.empty:
            return rows
        page = frame.to_dict(orient="records")
        new_count = 0
        for item in page:
            ric = item.get("RIC")
            if not isinstance(ric, str) or not ric:
                raise MarketDataError("search_missing_ric", "Search returned a constituent without a RIC")
            if ric not in seen:
                rows.append(item)
                seen.add(ric)
                new_count += 1
        if not new_count:
            raise MarketDataError("search_pagination_stalled", "Search pages repeat; refusing a silently truncated universe")
        if len(page) < _PAGE_SIZE:
            return rows
    raise MarketDataError("search_truncated", "Search window reached 10,000 rows; narrow expiry bounds before retrying")


def _base_filter(start, end, kind, low, high):
    return (
        "OptionStub eq 'SPXW.U' and UnderlyingQuoteRIC eq '.SPX' "
        f"and ExpiryDate ge {start.isoformat()} and ExpiryDate lt {end.isoformat()} "
        f"and CallPutOption eq '{kind}' and StrikePrice ge {low:.12g} and StrikePrice le {high:.12g}"
    )


def _verify_contract(item):
    underlying = item.get("UnderlyingQuoteRIC")
    underlyings = set(underlying) if isinstance(underlying, (list, tuple)) else {underlying}
    if item.get("OptionStub") != "SPXW.U" or underlyings != {".SPX"}:
        raise MarketDataError("unverified_contract", "Search constituent is not the verified SPXW / SPX family")
    currency = item.get("Currency")
    if currency is not None and str(currency) not in {"USD", "US Dollar", "US dollar"}:
        raise MarketDataError("unexpected_currency", "SPXW constituent currency is not USD")
    kind = str(item.get("CallPutOption", "")).lower()
    if kind not in {"call", "put"}:
        raise MarketDataError("unverified_contract", "Search did not identify Call/Put")
    return {
        "ric": item["RIC"], "expiry": _date(item["ExpiryDate"]).isoformat(),
        "expiry_source_value": str(item["ExpiryDate"]), "expiry_utc": None,
        "expiry_timestamp_status": "not_verified", "search_metadata": _safe_raw(item),
        "strike": _positive(item["StrikePrice"], "StrikePrice"), "kind": kind,
        "exercise_style": "European", "settlement": "PM", "settlement_type": "cash",
        "currency": "USD", "premium_unit": "index_points", "contract_multiplier": 100,
        "terms_source": _TERMS_SOURCE, "terms_verification": "verified_search_spxw_family",
    }


def _discover(session, names, as_of, spot, min_days, max_days):
    # Narrow strikes first to discover expiries without downloading a huge chain.
    discovered = {}
    start, stop = as_of + timedelta(days=min_days), as_of + timedelta(days=max_days + 1)
    while start < stop:
        end = min(start + timedelta(days=31), stop)
        for kind in ("Call", "Put"):
            for item in _search_pages(session, names, _base_filter(start, end, kind, 0.99 * spot, 1.01 * spot)):
                contract = _verify_contract(item)
                expiry = _date(contract["expiry"])
                if as_of + timedelta(days=min_days) <= expiry < stop:
                    discovered.setdefault(expiry, contract)
        start = end
    if not discovered:
        raise MarketDataError("options_not_found", "Search returned no eligible SPXW expiries; historical discovery and content permission are not verified")
    expiries = sorted(discovered)
    selected = set()
    for days in (7, 14, 30, 60, 90, 180, 365):
        if min_days <= days <= max_days:
            selected.add(min(expiries, key=lambda expiry: (abs((expiry - as_of).days - days), expiry)))
    if not selected:
        selected.add(expiries[0])
    contracts = []
    for expiry in sorted(selected):
        candidates = []
        for kind in ("Call", "Put"):
            rows = _search_pages(session, names, _base_filter(expiry, expiry + timedelta(days=1), kind, 0.8 * spot, 1.2 * spot))
            candidates.extend(_verify_contract(item) for item in rows)
        strikes = sorted({item["strike"] for item in candidates})
        chosen = {
            min(strikes, key=lambda strike: (abs(strike - spot * (0.8 + index * 0.02)), strike))
            for index in range(21)
        } if strikes else set()
        contracts.extend(item for item in candidates if item["strike"] in chosen)
    if not contracts:
        raise MarketDataError("options_not_found", "Selected expiries had no eligible SPXW contracts")
    return sorted({item["ric"]: item for item in contracts}.values(), key=lambda item: (item["expiry"], item["strike"], item["kind"]))


def _option_closes(session, contracts, as_of):
    accepted, rejected, raw = [], [], []
    by_ric = {item["ric"]: item for item in contracts}
    rics = list(by_ric)
    for offset in range(0, len(rics), 100):
        batch = rics[offset:offset + 100]
        try:
            response = _reference(session, batch, ["TR.PriceClose", "TR.PriceClose.date"], {
                "SDate": as_of.isoformat(), "EDate": as_of.isoformat(), "Frq": "D", "Curn": "USD",
            })
            frame = _frame(response, "SPXW historical closes")
            instrument = _column(frame, "Instrument", "RIC")
            price_column = _column(frame, "TR.PriceClose", "Price Close", "Close Price")
            date_column = _column(frame, "TR.PriceClose.date", "Date", "Price Close Date")
        except Exception as exc:
            raw.append({"endpoint": "fundamental_and_reference", "requested_rics": batch, "failure": _diagnostic(exc)})
            continue
        for _, row in frame.iterrows():
            ric = str(row[instrument])
            if ric not in by_ric:
                continue
            try:
                observed = _date(row[date_column])
                price = _positive(row[price_column], "option historical close")
                if observed != as_of:
                    raise MarketDataError("close_date_mismatch", "Option close is not dated as_of")
            except MarketDataError as exc:
                rejected.append({"ric": ric, "code": exc.code})
                continue
            accepted.append({
                **by_ric[ric], "close": price, "close_date": observed.isoformat(),
                "bid": None, "ask": None, "vendor_iv": None,
                "eligibility_basis": "historical_close_on_as_of",
                "close_field": "TR.PriceClose", "close_date_field": "TR.PriceClose.date",
                "close_source": "fundamental_and_reference.TR.PriceClose",
            })
        raw.append(_safe_raw(getattr(response.data, "raw", None)))
    accepted = list({item["ric"]: item for item in accepted}.values())
    seen = {item["ric"] for item in accepted}
    missing = [ric for ric in rics if ric not in seen]
    if missing:
        native, native_audit = _native_option_closes(session, missing, as_of)
        raw.extend(native_audit)
        for ric, value in native.items():
            values = value if isinstance(value, dict) else {"close": value, "bid": None, "ask": None}
            accepted.append({**by_ric[ric], **values, "close_date": as_of.isoformat(), "vendor_iv": None, "eligibility_basis": "historical_close_on_as_of", "close_field": "TRDPRC_1", "close_date_field": "DATE", "close_source": "historical_pricing.summaries.daily", "quote_freshness": "not_verified"})
        seen.update(native)
    # A rejected primary endpoint remains visible even if the documented native
    # endpoint succeeds; it is an audit event, not an older-day price fallback.
    if rejected:
        raw.append({"endpoint": "fundamental_and_reference", "primary_rejections": rejected.copy()})
    rejected = [item for item in rejected if item["ric"] not in seen]
    rejected.extend({"ric": ric, "code": "historical_close_missing"} for ric in rics if ric not in seen and not any(item["ric"] == ric for item in rejected))
    if not accepted:
        raise MarketDataError("option_history_unavailable", "No positive SPXW historical close matched as_of; inspect option history permissions and coverage", {"rejected_quotes": rejected})
    return accepted, rejected, raw


def _native_option_closes(session, rics, as_of):
    from lseg.data.content import historical_pricing

    closes, audit = {}, []
    for offset in range(0, len(rics), 100):
        batch = rics[offset:offset + 100]
        try:
            response = historical_pricing.summaries.Definition(
                universe=batch, interval=historical_pricing.Intervals.DAILY,
                start=as_of.isoformat(), end=as_of.isoformat(), fields=["TRDPRC_1", "BID", "ASK"],
            ).get_data(session=session)
            _check_response(response, "native SPXW historical daily summary")
            raw = getattr(getattr(response, "data", None), "raw", None)
            audit.append({"endpoint": "historical_pricing.summaries.daily", "requested_rics": batch, "response": _safe_raw(raw)})
            for obj in _objects(raw):
                ric = obj.get("universe", obj.get("instrument", obj.get("RIC")))
                if isinstance(ric, dict):
                    ric = ric.get("ric", ric.get("RIC"))
                if ric not in batch:
                    continue
                headers = obj.get("headers", [])
                names = [_norm(header.get("name", "") if isinstance(header, dict) else header) for header in headers]
                price_index = next((index for index, name in enumerate(names) if name == "trdprc1"), None)
                date_index = next((index for index, name in enumerate(names) if name in {"date", "datetime", "timestamp"}), None)
                bid_index = next((index for index, name in enumerate(names) if name == "bid"), None)
                ask_index = next((index for index, name in enumerate(names) if name == "ask"), None)
                if price_index is None or date_index is None:
                    continue
                for row in obj.get("data", []):
                    try:
                        if _date(row[date_index]) == as_of:
                            values = {"close": _positive(row[price_index], "native daily close"), "bid": None, "ask": None, "delay_status": (obj.get("qos") or {}).get("timeliness", "not_verified")}
                            for name, index in (("bid", bid_index), ("ask", ask_index)):
                                if index is not None:
                                    try:
                                        values[name] = _positive(row[index], name)
                                        values[name + "_date"] = as_of.isoformat()
                                    except (MarketDataError, IndexError, TypeError):
                                        pass
                            closes[ric] = values
                    except (MarketDataError, IndexError, KeyError, TypeError):
                        continue
        except Exception as exc:
            audit.append({"endpoint": "historical_pricing.summaries.daily", "requested_rics": batch, "failure": _diagnostic(exc)})
    return closes, audit


def _read_curves(path, as_of, expiries):
    required = {"as_of", "expiry", "discount_factor", "forward", "source"}
    rows = []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or []):
            raise MarketDataError("invalid_curves_csv", "Curves CSV requires as_of,expiry,discount_factor,forward,source")
        seen = set()
        for row in reader:
            observed, expiry = _date(row["as_of"]), _date(row["expiry"])
            if observed != as_of or expiry <= as_of:
                raise MarketDataError("curve_date_mismatch", "All curves must have the snapshot as_of and a future expiry")
            if expiry in seen:
                raise MarketDataError("duplicate_curve", "Curves CSV contains a duplicate expiry")
            seen.add(expiry)
            source = row["source"].strip()
            if not source:
                raise MarketDataError("invalid_curve_source", "Curves CSV requires an explicit source")
            rows.append({"as_of": observed.isoformat(), "expiry": expiry.isoformat(), "discount_factor": _positive(row["discount_factor"], "discount_factor"), "forward": _positive(row["forward"], "forward"), "source": source})
    missing = set(expiries) - {row["expiry"] for row in rows}
    if missing:
        raise MarketDataError("curve_expiry_missing", "Curves CSV lacks selected expiries: " + ", ".join(sorted(missing)))
    return sorted(rows, key=lambda row: row["expiry"])


def _ipa(session, as_of, expiries, *, forward_only=False):
    from lseg.data.delivery import endpoint_request

    body = {
        "outputs": ["DiscountCurve", "ForwardCurve", "Dividends", "UnderlyingSpot", "MoneynessStrike"],
        "universe": [{
            "underlyingType": "Eti", "surfaceTag": "SPXW-EOD-validation",
            "underlyingDefinition": {"instrumentCode": ".SPX"},
            "surfaceParameters": {"calculationDate": as_of.isoformat(), "timeStamp": "Close", "priceSide": "Last", "inputVolatilityType": "Implied", "volatilityModel": "SVI", "xAxis": "Date", "yAxis": "Strike", "usePriceFallbackLogic": False},
            "surfaceLayout": {"format": "Matrix", "xValues": sorted(expiries)},
        }],
    }
    if forward_only:
        definition = body["universe"][0]
        definition["surfaceParameters"].update({"yAxis": "Moneyness", "moneynessType": "Fwd"})
        definition["surfaceLayout"]["yValues"] = ["1.0"]
    response = endpoint_request.Definition(url=_SURFACE_URL, method="POST", body_parameters=body).get_data(session=session)
    _check_response(response, "IPA historical surface")
    data = getattr(response, "data", None)
    raw = getattr(data, "raw", None)
    if not isinstance(raw, (dict, list)):
        raise MarketDataError("ipa_no_data", "IPA did not return a structured surface response")
    for obj in _objects(raw):
        if str(obj.get("status", "")).lower() == "error":
            raise MarketDataError("ipa_content_error", "IPA returned an error status", _safe_raw(obj))
    # Record what we requested separately from vendor-supplied timestamps.  A
    # conflicting provider date will still invalidate the response below.
    raw = _safe_raw(raw)
    if not isinstance(raw, dict):
        raw = {"responses": raw}
    raw["request_context"] = {"as_of": as_of.isoformat(), "price_basis": "close", "moneyness_type": "Fwd" if forward_only else None, "atm_moneyness": 1.0 if forward_only else None, "x_axis": "Date", "y_axis": "Moneyness" if forward_only else "Strike"}
    return raw


def _objects(raw):
    if isinstance(raw, dict):
        yield raw
        for value in raw.values():
            yield from _objects(value)
    elif isinstance(raw, list):
        for value in raw:
            yield from _objects(value)


def _ipa_curves(raw, as_of, expiries):
    """Accept explicit same-date carry; log-linear interpolation stays in range."""
    discount, forward = {}, {}
    for obj in _objects(raw):
        for key, curve in obj.items():
            label = _norm(key)
            if label not in {"discountcurve", "forwardcurve"} or not isinstance(curve, dict):
                continue
            parameters = curve.get("curveParameters") or {}
            curve_date = curve.get("marketDataDate", parameters.get("marketDataDate"))
            if curve_date is None or _date(curve_date) != as_of:
                continue
            for point in curve.get("points", []):
                if not isinstance(point, dict):
                    continue
                expiry = point.get("endDate", point.get("expiry", point.get("date")))
                value = point.get("discountFactor") if label == "discountcurve" else point.get("forwardPrice", point.get("forward"))
                if expiry is None or value is None:
                    continue
                try:
                    (discount if label == "discountcurve" else forward)[_date(expiry).isoformat()] = _positive(value, label)
                except MarketDataError:
                    continue
    # The live IPA response supplies explicit forwardCurve.dataPoints.  Its
    # date is inherited only from independently checked enclosing provider
    # spot/discount dates and recorded request provenance.  MoneynessStrike
    # is not used: its units and expiry association are ambiguous.
    requested = any(isinstance(obj.get("request_context"), dict) and obj["request_context"].get("as_of") == as_of.isoformat() for obj in _objects(raw))
    for obj in _objects(raw):
        curve = obj.get("forwardCurve")
        if not isinstance(curve, dict) or not isinstance(curve.get("dataPoints"), dict):
            continue
        discount_curve = obj.get("discountCurve") or {}
        curve_date = (discount_curve.get("curveParameters") or {}).get("marketDataDate")
        spots = obj.get("underlyingSpot") or []
        if isinstance(spots, dict):
            spots = [spots]
        try:
            dated_spot = any(isinstance(item, dict) and item.get("instrumentCode") == ".SPX" and item.get("priceDate") is not None and _date(item["priceDate"]) == as_of for item in spots)
            explicit_date = obj.get("calculationDate", obj.get("marketDataDate"))
            if not requested or curve_date is None or _date(curve_date) != as_of or not dated_spot or (explicit_date is not None and _date(explicit_date) != as_of):
                continue
        except MarketDataError:
            continue
        for expiry, value in curve["dataPoints"].items():
            try:
                expiry = _date(expiry).isoformat()
                if _date(expiry) > as_of:
                    forward.setdefault(expiry, _positive(value, "explicit IPA forward"))
            except MarketDataError:
                continue
    rows = []
    for expiry in sorted(expiries):
        discount_value, discount_method = _interpolate_positive(discount, expiry)
        forward_value, forward_method = _interpolate_positive(forward, expiry)
        if discount_value is not None and forward_value is not None:
            rows.append({"as_of": as_of.isoformat(), "expiry": expiry, "discount_factor": discount_value, "forward": forward_value, "source": "LSEG IPA historical discount/forward points", "discount_interpolation": discount_method, "forward_interpolation": forward_method})
    return rows


def _interpolate_positive(points, target):
    if target in points:
        return points[target], "exact"
    dates = sorted((_date(key), value) for key, value in points.items())
    when = _date(target)
    for (left, value_left), (right, value_right) in zip(dates, dates[1:]):
        if left < when < right:
            weight = (when - left).days / (right - left).days
            return math.exp((1 - weight) * math.log(value_left) + weight * math.log(value_right)), "log_linear_ACT365F_in_range"
    return None, "outside_range"


def _evidence_hash(value):
    return hashlib.sha256(json.dumps(_safe_raw(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _surface_cells(raw, as_of):
    """Only a dated, SPX Date-by-Strike matrix is eligible for verification."""
    if not isinstance(raw, dict):
        return []
    context = raw.get("request_context") or {}
    if any(context.get(key) != expected for key, expected in (
        ("as_of", as_of.isoformat()), ("price_basis", "close"),
        ("x_axis", "Date"), ("y_axis", "Strike"),
    )):
        return []
    for obj in _objects(raw.get("data", [])):
        matrix = obj.get("surface")
        if not isinstance(matrix, list) or len(matrix) < 2 or not isinstance(matrix[0], list):
            continue
        curve = obj.get("discountCurve") or {}
        observed = (curve.get("curveParameters") or {}).get("marketDataDate")
        spots = obj.get("underlyingSpot") or []
        if isinstance(spots, dict):
            spots = [spots]
        try:
            spots = [spot for spot in spots if spot.get("instrumentCode") == ".SPX" and _date(spot.get("priceDate")) == as_of]
            provider_date = obj.get("calculationDate", obj.get("marketDataDate"))
            if observed is None or _date(observed) != as_of or len(spots) != 1 or (provider_date is not None and _date(provider_date) != as_of):
                continue
            spot = _positive(spots[0]["price"], "IPA spot")
            if spots[0].get("timeStamp", "Close") != "Close" or spots[0].get("priceSide", "Last") != "Last":
                continue
        except (MarketDataError, KeyError, TypeError):
            continue
        dividend_type = ((obj.get("dividends") or {}).get("curveDefinition") or {}).get("type")
        cells = []
        for row in matrix[1:]:
            if not isinstance(row, list) or len(row) != len(matrix[0]):
                continue
            try:
                expiry = _date(row[0])
                if expiry <= as_of:
                    continue
            except MarketDataError:
                continue
            for strike, value in zip(matrix[0][1:], row[1:]):
                try:
                    strike, value = _positive(strike, "surface strike"), float(value)
                    if not math.isfinite(value) or value < 0:
                        continue
                except (MarketDataError, TypeError, ValueError):
                    continue
                cells.append({"expiry": expiry.isoformat(), "strike": strike, "surface_value": value,
                              "spot": spot, "discount_curve_id": curve.get("marketDataId"),
                              "dividend_type": dividend_type})
        return cells
    return []


def _financial_rows(raw):
    for obj in _objects(raw):
        headers, values = obj.get("headers"), obj.get("data")
        if not isinstance(headers, list) or not isinstance(values, list):
            continue
        names = [header.get("name", "") if isinstance(header, dict) else str(header) for header in headers]
        for row in values:
            if isinstance(row, list) and len(row) == len(names):
                yield dict(zip(names, row))
            elif isinstance(row, dict):
                yield row


def _black_call(spot, strike, expiry, as_of, discount, forward, iv):
    t = (_date(expiry) - as_of).days / 365.0
    sd = iv * math.sqrt(t)
    if sd == 0:
        return discount * max(forward - strike, 0.0)
    d1 = math.log(forward / strike) / sd + sd / 2
    d2 = d1 - sd
    cdf = lambda x: .5 * math.erfc(-x / math.sqrt(2.0))
    return discount * (forward * cdf(d1) - strike * cdf(d2))


def _assess_surface_evidence(raw, as_of, curves, requests, responses, *, absolute_percent=1e-6, relative=1e-9, price_tolerance=0.01, typed_dataset=False):
    """Bind typed FinancialContracts outputs to individual matrix cells.

    The unit is established by numeric identity to the documented percentage
    field, rather than by a heuristic volatility threshold. EURO and the
    returned BlackScholes model establish the convention. A separate Black
    price identity checks compatibility with the captured discount and forward.
    """
    cells = _surface_cells(raw, as_of)
    curve_map = {curve["expiry"]: curve for curve in curves if curve.get("as_of") == as_of.isoformat()}
    wanted = {}
    for request in requests:
        for instrument in request.get("universe", []):
            definition = instrument.get("instrumentDefinition") or {}
            wanted[definition.get("instrumentTag")] = (definition, instrument.get("pricingParameters") or {})
    rows = {}
    duplicates = set()
    for response in responses:
        for row in _financial_rows(response):
            tag = row.get("InstrumentTag")
            if tag in rows:
                duplicates.add(tag)
            rows[tag] = row
    by_key = {(cell["expiry"], cell["strike"]): cell for cell in cells}
    accepted, diagnostics, unit_candidates = [], [], []
    for tag, (definition, parameters) in wanted.items():
        try:
            expiry, strike = _date(definition["endDate"]).isoformat(), float(definition["strike"])
            cell = by_key[(expiry, strike)]
            row = rows[tag]
            if tag in duplicates:
                raise ValueError("duplicate_response_tag")
            error_message = str(row.get("ErrorMessage") or "").strip()
            if row.get("ErrorCode") not in (None, 0, "0", "") or error_message:
                raise ValueError("provider_point_error: " + str(row.get("ErrorMessage") or row.get("ErrorCode")))
            for field, expected in (("VolatilityType", "SVISurface"), ("PricingModelType", "BlackScholes"),
                                    ("ExerciseStyle", "EURO"), ("OptionType", "Vanilla"),
                                    ("UnderlyingRIC", ".SPX"), ("CallPut", "Call")):
                matches = row.get(field) == expected if field == "UnderlyingRIC" else _norm(row.get(field)) == _norm(expected)
                if not matches:
                    raise ValueError("unexpected_" + field)
            for field in ("ValuationDate", "MarketDataDate"):
                if _date(row.get(field)) != as_of:
                    raise ValueError("date_mismatch")
            if _date(row.get("EndDate")).isoformat() != expiry or not math.isclose(float(row["Strike"]), strike, rel_tol=1e-12, abs_tol=1e-8):
                raise ValueError("point_mismatch")
            if not math.isclose(float(row["UnderlyingPrice"]), cell["spot"], rel_tol=1e-12, abs_tol=1e-8):
                raise ValueError("spot_mismatch")
            explicit_spot = parameters.get("underlyingPrice") if typed_dataset else None
            if explicit_spot is not None:
                if not math.isclose(float(explicit_spot), cell["spot"], rel_tol=1e-12, abs_tol=1e-8) or _norm(row.get("UnderlyingPriceSide")) != "user":
                    raise ValueError("captured_spot_override_not_confirmed")
            elif _norm(row.get("UnderlyingTimeStamp")) != "close" or _norm(row.get("UnderlyingPriceSide")) != "last":
                raise ValueError("underlying_close_not_confirmed")
            expected_dividend_type = parameters.get("dividendType") if typed_dataset else cell.get("dividend_type")
            if not expected_dividend_type or _norm(row.get("DividendType")) != _norm(expected_dividend_type):
                raise ValueError("dividend_type_mismatch")
            typed = float(row["VolatilityPercent"])
            if not math.isfinite(typed) or typed < 0:
                raise ValueError("invalid_typed_volatility")
            if typed_dataset:
                # This is a separate, fully typed supplier dataset. The
                # untyped matrix supplies grid locations, not IV values.
                unit = "percent"
            else:
                candidates = {unit for unit, factor in (("percent", 1), ("decimal", 100))
                              if math.isclose(cell["surface_value"] * factor, typed,
                                              rel_tol=relative, abs_tol=absolute_percent)}
                if len(candidates) != 1:
                    raise ValueError("unit_identity_ambiguous_or_mismatch")
                unit = candidates.pop()
            curve = curve_map[expiry]
            if typed_dataset:
                tenor = (_date(expiry) - as_of).days / 365.0
                for field, expected in (("riskFreeRatePercent", -100 * math.log(curve["discount_factor"]) / tenor),
                                        ("dividendYieldPercent", -100 * math.log(curve["forward"] * curve["discount_factor"] / cell["spot"]) / tenor)):
                    if not math.isclose(float(parameters.get(field)), expected, rel_tol=1e-10, abs_tol=1e-8) or not math.isclose(float(row.get(field[0].upper() + field[1:])), expected, rel_tol=1e-10, abs_tol=1e-8):
                        raise ValueError("carry_override_not_confirmed")
            expected_price = _black_call(cell["spot"], strike, expiry, as_of, curve["discount_factor"], curve["forward"], typed / 100)
            premium = float(row["MarketValueInDealCcy"])
            if not math.isfinite(premium) or not math.isclose(premium, expected_price, rel_tol=relative, abs_tol=price_tolerance):
                raise ValueError("carry_price_identity_mismatch")
            accepted.append({"expiry": expiry, "strike": strike, "iv": typed / 100,
                             "surface_value": cell["surface_value"], "typed_volatility_percent": typed,
                             "instrument_tag": tag, "discount_factor": curve["discount_factor"], "forward": curve["forward"],
                             "financial_contract_price": premium, "black_price": expected_price,
                             "discount_curve_id": row.get("DiscountCurveId"), "dividend_type": row["DividendType"],
                             "captured_dividend_type": cell["dividend_type"],
                             "spot_basis": "explicit dated captured IPA underlyingSpot override" if explicit_spot is not None else "provider returned Close Last",
                             "matrix_value_verified": not typed_dataset})
            unit_candidates.append(unit)
        except (ValueError, TypeError, KeyError, MarketDataError) as exc:
            diagnostics.append({"instrument_tag": tag, "code": str(exc)[:100]})
    units = set(unit_candidates)
    if len(units) > 1:
        diagnostics.append({"code": "inconsistent_surface_units"})
        accepted = []
    entire = bool(cells) and len(accepted) == len(cells)
    return {"schema_version": 1, "method": "ipa_financial_contracts_typed_surface" if typed_dataset else "ipa_financial_contracts_typed_fields",
            "status": "verified" if entire else "partial" if accepted else "unverified",
            "as_of": as_of.isoformat(), "surface_sha256": _evidence_hash(raw),
            "unit": next(iter(units)) if len(units) == 1 else None,
            "typed_field": "VolatilityPercent", "pricing_model_type": "BlackScholes",
            "exercise_style": "EURO", "volatility_type": "SVISurface", "price_basis": "close",
            "matrix_values_verified": not typed_dataset and entire,
            "dataset_basis": "FinancialContracts SVISurface on the saved matrix grid" if typed_dataset else "saved matrix matched to FinancialContracts",
            "carry_basis": "explicit captured IPA forward/discount overrides checked against returned rates and Black premium" if typed_dataset else "Black premium identity against captured IPA forward/discount",
            "coverage": {"total": len(cells), "requested": len(wanted), "verified": len(accepted), "entire_surface_verified": entire},
            "tolerance": {"absolute_percent": absolute_percent, "relative": relative, "absolute_price_index_points": price_tolerance},
            "official_sources": [_CONTRACT_DOC, _SURFACE_DOC], "points": accepted, "diagnostics": diagnostics,
            "requests": _safe_raw(requests), "responses": _safe_raw(responses),
            "request_sha256": _evidence_hash(requests), "response_sha256": _evidence_hash(responses)}


def _ipa_surface_evidence(session, raw, as_of, curves, *, limit=None, typed_dataset=False, dividend_type=None):
    """Verify all Date x Strike cells through typed IPA FinancialContracts.

    Set limit for a permission probe only; a partial result labels its actual
    coverage and never verifies the unqueried cells. No credential is recorded.
    """
    from lseg.data.delivery import endpoint_request

    cells = _surface_cells(raw, as_of)
    if not cells:
        raise MarketDataError("surface_not_verifiable", "No dated SPX Date x Strike matrix is available")
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer or None")
        # Probe near-spot points across expiries before expensive wing queries.
        cells = sorted(cells, key=lambda cell: (abs(math.log(cell["strike"] / cell["spot"])), cell["expiry"]))[:limit]
    fields = ["InstrumentTag", "EndDate", "Strike", "CallPut", "OptionType", "ExerciseStyle",
              "UnderlyingRIC", "UnderlyingPrice", "UnderlyingTimeStamp", "UnderlyingPriceSide",
              "ValuationDate", "MarketDataDate", "PricingModelType", "VolatilityType", "VolatilityPercent",
              "DiscountCurveId", "DividendType", "RiskFreeRatePercent", "DividendYieldPercent",
              "MarketValueInDealCcy", "YearsToExpiry", "ErrorCode", "ErrorMessage"]
    requests, responses = [], []
    curve_map = {curve["expiry"]: curve for curve in curves if curve.get("as_of") == as_of.isoformat()}
    for offset in range(0, len(cells), 100):
        universe = []
        for index, cell in enumerate(cells[offset:offset + 100], offset):
            if cell["expiry"] not in curve_map:
                raise MarketDataError("surface_carry_missing", "Typed verification needs captured same-date discount and forward for every requested expiry")
            curve = curve_map[cell["expiry"]]
            tenor = (_date(cell["expiry"]) - as_of).days / 365.0
            # Historical ImpliedYield retrieval can be unavailable in Financial
            # Contracts. Explicitly carry the captured curve into this call;
            # the returned Black price identity verifies these conventions.
            risk_free_percent = -100 * math.log(curve["discount_factor"]) / tenor
            dividend_percent = -100 * math.log(curve["forward"] * curve["discount_factor"] / cell["spot"]) / tenor
            universe.append({"instrumentType": "Option", "instrumentDefinition": {
                "instrumentTag": f"SPX-surface-{index}", "underlyingType": "Eti",
                "underlyingDefinition": {"instrumentCode": ".SPX"}, "strike": cell["strike"],
                "endDate": cell["expiry"], "exerciseStyle": "EURO", "callPut": "Call", "buySell": "Buy",
                "lotSize": 1, "dealContract": 1,
            }, "pricingParameters": {
                "valuationDate": as_of.isoformat(), "marketDataDate": as_of.isoformat(),
                "pricingModelType": "BlackScholes", "volatilityType": "SVISurface",
                "optionTimeStamp": "Close", "optionPriceSide": "Last",
                "underlyingTimeStamp": "Close", "underlyingPriceSide": "Last",
                "dividendType": dividend_type or cell["dividend_type"], "reportCcy": "USD",
                "riskFreeRatePercent": risk_free_percent, "dividendYieldPercent": dividend_percent,
            }})
            if typed_dataset:
                # The native summary and surface entitlement can succeed even
                # when FinancialContracts lacks its own .SPX history access.
                # Supply the already dated captured spot; never call it a
                # newly returned market quote in the verification metadata.
                universe[-1]["pricingParameters"]["underlyingPrice"] = cell["spot"]
        body = {"universe": universe, "fields": fields, "outputs": ["Data", "Headers"]}
        requests.append(body)
        response = endpoint_request.Definition(url=_CONTRACT_URL, method="POST", body_parameters=body).get_data(session=session)
        _check_response(response, "IPA typed surface verification")
        responses.append(_safe_raw(getattr(getattr(response, "data", None), "raw", None)))
    return _assess_surface_evidence(raw, as_of, curves, requests, responses, typed_dataset=typed_dataset)


def _ipa_surface_list_probe(session, raw, as_of, *, limit=None):
    """Request the exact saved grid in List format to discover typed headers.

    This is evidence collection only: no unit or pricing convention is inferred
    from an unspecified header. Surface parameters match the matrix request.
    """
    from lseg.data.delivery import endpoint_request

    cells = _surface_cells(raw, as_of)
    if not cells:
        raise MarketDataError("surface_not_verifiable", "No dated SPX Date x Strike matrix is available")
    if limit is not None:
        cells = sorted(cells, key=lambda cell: (abs(math.log(cell["strike"] / cell["spot"])), cell["expiry"]))[:limit]
    body = {
        "outputs": ["DiscountCurve", "ForwardCurve", "Dividends", "UnderlyingSpot"],
        "universe": [{"underlyingType": "Eti", "surfaceTag": "SPXW-EOD-validation",
                      "underlyingDefinition": {"instrumentCode": ".SPX"},
                      "surfaceParameters": {"calculationDate": as_of.isoformat(), "timeStamp": "Close",
                                            "priceSide": "Last", "inputVolatilityType": "Implied",
                                            "volatilityModel": "SVI", "xAxis": "Date", "yAxis": "Strike",
                                            "usePriceFallbackLogic": False},
                      "surfaceLayout": {"format": "List", "dataPoints": [
                          {"x": cell["expiry"], "y": str(cell["strike"])} for cell in cells]}}],
    }
    response = endpoint_request.Definition(url=_SURFACE_URL, method="POST", body_parameters=body).get_data(session=session)
    _check_response(response, "IPA exact-grid List metadata probe")
    value = _safe_raw(getattr(getattr(response, "data", None), "raw", None))
    return {"request": body, "response": value, "request_sha256": _evidence_hash(body),
            "response_sha256": _evidence_hash(value), "matrix_sha256": _evidence_hash(raw),
            "as_of": as_of.isoformat(), "official_source": _SURFACE_DOC}


def _vendor_surface(raw, as_of, curves, *, verification=None):
    """Only explicit provider metadata or replayable typed evidence is accepted."""
    if verification is not None:
        if verification.get("as_of") != as_of.isoformat() or verification.get("surface_sha256") != _evidence_hash(raw):
            return None
        if _evidence_hash(verification.get("requests", [])) != verification.get("request_sha256") or _evidence_hash(verification.get("responses", [])) != verification.get("response_sha256"):
            return None
        tolerance = verification.get("tolerance") or {}
        # Recompute from preserved typed outputs; do not trust supplied flags.
        evidence = _assess_surface_evidence(raw, as_of, curves, verification.get("requests", []), verification.get("responses", []),
                                          absolute_percent=min(float(tolerance.get("absolute_percent", 1e-6)), 1e-6),
                                          relative=min(float(tolerance.get("relative", 1e-9)), 1e-9),
                                          price_tolerance=min(float(tolerance.get("absolute_price_index_points", .01)), .01),
                                          typed_dataset=verification.get("method") == "ipa_financial_contracts_typed_surface")
        if not evidence["points"]:
            return None
        summary = {key: value for key, value in evidence.items() if key not in {"requests", "responses", "points"}}
        return {"as_of": as_of.isoformat(), "volatility_unit": "decimal", "convention": "black",
                "points": [{key: point[key] for key in ("expiry", "strike", "iv", "forward", "discount_factor")} for point in evidence["points"]],
                "metadata_basis": "IPA_typed_fields_verification", "source": "LSEG IPA FinancialContracts SVISurface" if evidence["method"] == "ipa_financial_contracts_typed_surface" else "LSEG IPA", "independent_reference": False,
                "units_verified_by_response": True, "verification": summary}
    # Some future provider schema may return fully explicit point metadata.
    for obj in _objects(raw):
        if str(obj.get("volatilityUnit", "")).lower() not in {"decimal", "percent", "percentage"} or str(obj.get("convention", "")).lower() not in {"black", "blackscholes"}:
            continue
        try:
            if _date(obj.get("as_of", obj.get("calculationDate"))) != as_of:
                continue
        except MarketDataError:
            continue
        points = []
        divisor = 1 if obj["volatilityUnit"].lower() == "decimal" else 100
        for point in obj.get("points", []):
            try:
                expiry = _date(point["expiry"]).isoformat()
                value = float(point["iv"]) / divisor
                if not math.isfinite(value) or value < 0 or _date(expiry) <= as_of:
                    continue
                points.append({"expiry": expiry, "strike": _positive(point["strike"], "vendor strike"), "iv": value})
            except (MarketDataError, KeyError, TypeError, ValueError):
                continue
        if points:
            return {"as_of": as_of.isoformat(), "volatility_unit": "decimal", "convention": "black", "points": points,
                    "metadata_basis": "explicit_provider_response", "source": "LSEG IPA", "units_verified_by_response": True, "independent_reference": False}
    return None


def doctor(*, prompt=False):
    """Report session, spot, Search, option history, and IPA independently."""
    report = {"provider": "LSEG", "session_type": "desktop", "checks": {name: {"status": "not_run"} for name in ("session", "spot", "search", "history", "ipa")}}
    try:
        with _desktop(prompt=prompt) as (session, ld, key):
            report["library_version"] = getattr(ld, "__version__", "unknown")
            report["checks"]["session"] = {"status": "ok"}
            observed, spot = None, None
            try:
                observed, spot, _ = _underlying_close(session)
                report["checks"]["spot"] = {"status": "ok", "ric": ".SPX", "as_of": observed.isoformat(), "close": spot}
            except Exception as exc:
                report["checks"]["spot"] = _diagnostic(exc, key)
            try:
                names = _search_metadata(session)
                if observed is not None:
                    sample = _search_pages(session, names, _base_filter(observed + timedelta(days=7), observed + timedelta(days=15), "Call", 0.99 * spot, 1.01 * spot))
                    if not sample:
                        raise MarketDataError("options_not_found", "Search metadata works but no SPXW sample was found")
                    contract = _verify_contract(sample[0])
                    report["checks"]["search"] = {"status": "ok", "sample_ric": contract["ric"], "metadata_verified": True}
                    try:
                        quotes, _, _ = _option_closes(session, [contract], observed)
                        report["checks"]["history"] = {"status": "ok", "as_of": observed.isoformat(), "sample_ric": quotes[0]["ric"]}
                    except Exception as exc:
                        report["checks"]["history"] = _diagnostic(exc, key)
                else:
                    report["checks"]["search"] = {"status": "metadata_only", "metadata_verified": True}
            except Exception as exc:
                report["checks"]["search"] = _diagnostic(exc, key)
            if observed is not None:
                try:
                    raw = _ipa(session, observed, [(observed + timedelta(days=30)).isoformat()])
                    has_surface = any(isinstance(obj.get("surface"), list) and len(obj["surface"]) > 1 for obj in _objects(raw))
                    if not has_surface:
                        raise MarketDataError("ipa_surface_missing", "IPA responded without a recognizable surface; inspect returned statuses")
                    report["checks"]["ipa"] = {"status": "ok", "as_of": observed.isoformat(), "surface_units_verified": False}
                except Exception as exc:
                    report["checks"]["ipa"] = _diagnostic(exc, key)
    except Exception as exc:
        report["checks"]["session"] = _diagnostic(exc)
    statuses = {name: check["status"] for name, check in report["checks"].items()}
    report["status"] = "error" if statuses["session"] != "ok" else ("ok" if all(value == "ok" for value in statuses.values()) else "partial")
    return report


def fetch_snapshot(*, as_of=None, curves_path=None, min_days=7, max_days=365, output=None, verify_vendor_surface=False):
    """Capture historical SPXW closes and preserve a needs_curves snapshot if needed."""
    if isinstance(min_days, bool) or isinstance(max_days, bool) or not isinstance(min_days, int) or not isinstance(max_days, int) or not 1 <= min_days <= max_days <= 3650:
        raise MarketDataError("invalid_expiry_window", "Expiry bounds must be integers with 1 <= min_days <= max_days <= 3650")
    if not isinstance(verify_vendor_surface, bool):
        raise ValueError("verify_vendor_surface must be a boolean")
    with _desktop() as (session, ld, key):
        observed, spot, spot_raw = _underlying_close(session, as_of)
        names = _search_metadata(session)
        contracts = _discover(session, names, observed, spot, min_days, max_days)
        quotes, rejected, quote_raw = _option_closes(session, contracts, observed)
        expiries = {item["expiry"] for item in quotes}
        diagnostics = {"rejected_quotes": rejected}
        ipa_raw, forward_raw, curves = None, None, []
        try:
            ipa_raw = _ipa(session, observed, expiries)
            curves = _ipa_curves(ipa_raw, observed, expiries)
            diagnostics["ipa"] = {"status": "received", "surface_units_verified": False}
            if {curve["expiry"] for curve in curves} < expiries:
                try:
                    forward_raw = _ipa(session, observed, expiries, forward_only=True)
                    curves = _ipa_curves({"responses": [ipa_raw, forward_raw]}, observed, expiries)
                    diagnostics["forward_ipa"] = {"status": "received"}
                except Exception as exc:
                    diagnostics["forward_ipa"] = _diagnostic(exc, key)
        except Exception as exc:
            diagnostics["ipa"] = _diagnostic(exc, key)
        verification, vendor_surface = None, None
        if ipa_raw is not None and verify_vendor_surface:
            try:
                verification = _ipa_surface_evidence(session, ipa_raw, observed, curves, typed_dataset=True, dividend_type="HistoricalYield")
                vendor_surface = _vendor_surface(ipa_raw, observed, curves, verification=verification)
                diagnostics["vendor_surface_verification"] = {"status": verification["status"], "coverage": verification["coverage"], "diagnostics": verification["diagnostics"]}
            except Exception as exc:
                diagnostics["vendor_surface_verification"] = _diagnostic(exc, key)
        elif ipa_raw is not None:
            vendor_surface = _vendor_surface(ipa_raw, observed, curves)
            diagnostics["vendor_surface_verification"] = {"status": "not_requested", "message": "Typed FinancialContracts verification is optional and requires its own content access; enable verify_vendor_surface to capture evidence"}
        diagnostics["vendor_surface"] = {"status": "ok" if vendor_surface else "unavailable", "metadata_basis": vendor_surface.get("metadata_basis") if vendor_surface else None, "message": "No compatible verified same-date vendor surface" if not vendor_surface else "Supplier curve verified through typed FinancialContracts responses; inspect coverage"}
        if curves_path is not None:
            curves = _read_curves(curves_path, observed, expiries)
        complete_curves = {item["expiry"] for item in curves} >= expiries
        if not complete_curves:
            diagnostics["missing_curve_expiries"] = sorted(expiries - {item["expiry"] for item in curves})
            diagnostics["curves"] = {"status": "needs_curves", "message": "Supply a dated curves CSV; no zero rate or dividend assumptions were made"}
        snapshot = {
            "schema_version": 1,
            "source": {
                "provider": "LSEG", "as_of": observed.isoformat(), "price_basis": "close",
                "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                "library_version": getattr(ld, "__version__", "unknown"), "session_type": "desktop",
                "underlying_ric": ".SPX", "option_family": "SPXW.U",
                "discovery_basis": "current_search_index", "historical_survivorship_risk": True,
                "eligibility_basis": "positive_historical_close_on_as_of",
                "delay_status": next(iter({quote.get("delay_status", "not_verified") for quote in quotes})) if len({quote.get("delay_status", "not_verified") for quote in quotes}) == 1 else "mixed",
                "quote_freshness": "not_verified", "expiry_timestamp_status": "not_verified", "terms_source": _TERMS_SOURCE,
                "validation_status": "ready" if complete_curves else "needs_curves",
            },
            "spot": spot, "curves": curves, "quotes": quotes, "vendor_surface": vendor_surface,
            "diagnostics": diagnostics,
            "raw_responses": {"underlying_close": spot_raw, "search_contracts": [item["search_metadata"] for item in contracts], "option_closes": quote_raw, "ipa": ipa_raw, "forward_ipa": forward_raw, "vendor_surface_verification": verification},
        }
    if output is not None:
        path = Path(output)
        if path.suffix.lower() != ".json":
            path = path / "snapshot.json"
        from .snapshot import write_json

        raw = snapshot.pop("raw_responses")
        raw_path = path.with_name("raw_responses.json")
        write_json(raw_path, raw)
        snapshot["source"]["raw_response_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        snapshot["source"]["raw_response_file"] = raw_path.name
        write_json(path, snapshot)
    return snapshot
