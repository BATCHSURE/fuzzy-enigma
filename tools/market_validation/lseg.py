"""Bounded, historical SPXW snapshots through an explicit LSEG Desktop session.

The optional LSEG dependency is imported only when a request is made.  Credentials
come from LSEG_APP_KEY (or doctor's hidden prompt), never a configuration file.
"""

from __future__ import annotations

import csv
import getpass
import hashlib
import importlib
import math
import os
import re
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


_TERMS_SOURCE = "https://www.cboe.com/tradable-products/sp-500/spx-options/spx-specifications"
_SURFACE_URL = "https://api.refinitiv.com/data/quantitative-analytics-curves-and-surfaces/v1/surfaces"
_SDK_SOURCE = "https://pypi.org/project/lseg-data/2.1.1/#files"
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


def _vendor_surface(raw, as_of, curves):
    """Normalize explicit units or a labeled SDK assumption; never guess by size."""
    curve_map = {curve["expiry"]: curve for curve in curves}
    inferred_objects = set()
    matrix_axes = {}
    for response in _objects(raw):
        context = response.get("request_context") or {}
        for candidate in _objects(response.get("data", [])):
            if context.get("y_axis"):
                matrix_axes[id(candidate)] = context["y_axis"]
        if context.get("as_of") != as_of.isoformat() or context.get("price_basis") != "close" or context.get("x_axis") != "Date" or context.get("y_axis") != "Strike":
            continue
        for candidate in _objects(response.get("data", [])):
            discount_date = ((candidate.get("discountCurve") or {}).get("curveParameters") or {}).get("marketDataDate")
            spots = candidate.get("underlyingSpot") or []
            if isinstance(spots, dict):
                spots = [spots]
            try:
                dated_spot = any(isinstance(spot, dict) and spot.get("instrumentCode") == ".SPX" and spot.get("priceDate") is not None and _date(spot["priceDate"]) == as_of for spot in spots)
                provider_date = candidate.get("calculationDate", candidate.get("marketDataDate"))
                if discount_date is not None and _date(discount_date) == as_of and dated_spot and (provider_date is None or _date(provider_date) == as_of):
                    inferred_objects.add(id(candidate))
            except MarketDataError:
                continue
    for obj in _objects(raw):
        fields = {_norm(key): value for key, value in obj.items()}
        unit = str(fields.get("volatilityunit", "")).lower()
        convention = str(fields.get("convention", fields.get("volatilityconvention", fields.get("pricingmodel", "")))).lower()
        observed = fields.get("calculationdate", fields.get("asof", fields.get("marketdatadate")))
        assumed = id(obj) in inferred_objects and not unit and not convention
        if assumed:
            # This is a displayed vendor benchmark with an explicit SDK-based
            # convention assumption, never an independent pricing reference.
            unit, convention, observed = "percent", "black", as_of
        if unit not in {"decimal", "percent", "percentage"} or convention not in {"black", "blackscholes", "black-scholes"} or observed is None:
            continue
        try:
            if _date(observed) != as_of:
                continue
        except MarketDataError:
            continue
        divisor = 100.0 if unit in {"percent", "percentage"} else 1.0
        points = []
        candidates = obj.get("points", [])
        matrix = obj.get("surface")
        if isinstance(matrix, list) and len(matrix) > 1 and isinstance(matrix[0], list):
            if matrix_axes.get(id(obj), "Strike") != "Strike":
                continue
            candidates = []
            for row in matrix[1:]:
                if not isinstance(row, list) or len(row) != len(matrix[0]):
                    continue
                for column, iv in zip(matrix[0][1:], row[1:]):
                    # Matrix axes must be the requested Date x Strike, or an
                    # explicitly recognizable transposition of those axes.
                    try:
                        expiry, strike = _date(row[0]), _positive(column, "vendor strike")
                    except MarketDataError:
                        try:
                            expiry, strike = _date(column), _positive(row[0], "vendor strike")
                        except MarketDataError:
                            continue
                    candidates.append({"expiry": expiry.isoformat(), "strike": strike, "iv": iv})
        for point in candidates:
            if not isinstance(point, dict):
                continue
            try:
                expiry = _date(point["expiry"]).isoformat()
                strike = _positive(point["strike"], "vendor strike")
                iv = float(point["iv"]) / divisor
                if not math.isfinite(iv) or iv < 0 or _date(expiry) <= as_of:
                    continue
            except (MarketDataError, KeyError, TypeError, ValueError):
                continue
            normalized = {"expiry": expiry, "strike": strike, "iv": iv}
            # Attach carry only when it came from this IPA response, allowing
            # the reporting layer to verify compatibility with supplied CSV.
            if expiry in curve_map:
                normalized.update({name: curve_map[expiry][name] for name in ("forward", "discount_factor")})
            points.append(normalized)
        if points:
            result = {"as_of": as_of.isoformat(), "volatility_unit": "decimal", "convention": "black", "points": points, "metadata_basis": "SDK_example_convention" if assumed else "explicit_provider_response", "source": "LSEG IPA", "independent_reference": False, "units_verified_by_response": not assumed}
            if assumed:
                result.update({"unit_assumption": "percent", "convention_assumption": "Black for European index implied volatility", "convention_source": _SDK_SOURCE, "convention_source_module": "lseg.data.content.ipa.surfaces._surfaces_data_provider.parse_axis docstring, lseg-data 2.1.1"})
            return result
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


def fetch_snapshot(*, as_of=None, curves_path=None, min_days=7, max_days=365, output=None):
    """Capture historical SPXW closes and preserve a needs_curves snapshot if needed."""
    if isinstance(min_days, bool) or isinstance(max_days, bool) or not isinstance(min_days, int) or not isinstance(max_days, int) or not 1 <= min_days <= max_days <= 3650:
        raise MarketDataError("invalid_expiry_window", "Expiry bounds must be integers with 1 <= min_days <= max_days <= 3650")
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
        vendor_surface = _vendor_surface(ipa_raw, observed, curves) if ipa_raw is not None else None
        diagnostics["vendor_surface"] = {"status": "ok" if vendor_surface else "unavailable", "metadata_basis": vendor_surface.get("metadata_basis") if vendor_surface else None, "message": "No compatible same-date vendor surface" if not vendor_surface else "Displayed vendor benchmark; inspect convention assumptions before comparison"}
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
            "raw_responses": {"underlying_close": spot_raw, "search_contracts": [item["search_metadata"] for item in contracts], "option_closes": quote_raw, "ipa": ipa_raw, "forward_ipa": forward_raw},
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
