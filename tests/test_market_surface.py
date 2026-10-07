"""Offline supplier unit/convention verification; no Desktop session is opened."""

from copy import deepcopy
from datetime import date
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.market_validation.lseg import (
    _assess_surface_evidence, _evidence_hash, _surface_cells, _vendor_surface,
)

AS_OF = date(2026, 10, 6)
EXPIRY = "2027-10-06"


def fixture(*, decimal=False):
    """Known typed-field schema and prices independently pinned with QuantLib.

    QuantLib AnalyticEuropeanEngine: S=F=100, D=1, sigma=.2, ACT/365F
    exactly one year, European calls. This fixture models the provider's
    documented field contract; it is not a claimed captured market quote.
    """
    value = .2 if decimal else 20.0
    raw = {
        "request_context": {"as_of": AS_OF.isoformat(), "price_basis": "close", "x_axis": "Date", "y_axis": "Strike"},
        "data": [{"discountCurve": {"marketDataId": "fixture-USD-curve", "curveParameters": {"marketDataDate": AS_OF.isoformat()}},
                  "dividends": {"curveDefinition": {"type": "ImpliedYield"}},
                  "underlyingSpot": [{"instrumentCode": ".SPX", "price": 100, "priceDate": AS_OF.isoformat()}],
                  "surface": [[None, "100", "110"], [EXPIRY, value, value]]}],
    }
    curves = [{"as_of": AS_OF.isoformat(), "expiry": EXPIRY, "discount_factor": 1, "forward": 100, "source": "fixture"}]
    definitions, rows = [], []
    prices = {100: 7.9655674554058, 110: 4.292010941409885}
    for index, strike in enumerate((100, 110)):
        tag = f"SPX-surface-{index}"
        definitions.append({"instrumentType": "Option", "instrumentDefinition": {
            "instrumentTag": tag, "strike": strike, "endDate": EXPIRY,
            "exerciseStyle": "EURO", "callPut": "Call", "underlyingDefinition": {"instrumentCode": ".SPX"},
        }, "pricingParameters": {"volatilityType": "SVISurface", "pricingModelType": "BlackScholes"}})
        rows.append({"InstrumentTag": tag, "EndDate": EXPIRY, "Strike": strike,
                     "CallPut": "Call", "OptionType": "Vanilla", "ExerciseStyle": "EURO",
                     "UnderlyingRIC": ".SPX", "UnderlyingPrice": 100, "UnderlyingTimeStamp": "Close",
                     "UnderlyingPriceSide": "Last", "ValuationDate": AS_OF.isoformat(),
                     "MarketDataDate": AS_OF.isoformat(), "PricingModelType": "BlackScholes",
                     "VolatilityType": "SVISurface", "VolatilityPercent": 20,
                     "DiscountCurveId": "fixture-USD-curve", "DividendType": "ImpliedYield",
                     "MarketValueInDealCcy": prices[strike], "ErrorCode": 0, "ErrorMessage": None})
    fields = list(rows[0])
    responses = [{"headers": [{"name": name} for name in fields], "data": [[row[name] for name in fields] for row in rows]}]
    return raw, curves, [{"universe": definitions}], responses


def update_field(responses, field, value, *, row=0):
    index = next(i for i, header in enumerate(responses[0]["headers"]) if header["name"] == field)
    responses[0]["data"][row][index] = value


class SupplierSurfaceTests(unittest.TestCase):
    def test_ambiguous_matrix_has_no_assumed_overlay(self):
        raw, curves, _, _ = fixture()
        self.assertEqual(len(_surface_cells(raw, AS_OF)), 2)
        self.assertIsNone(_vendor_surface(raw, AS_OF, curves))

    def test_documented_percent_identity_and_provenance(self):
        raw, curves, requests, responses = fixture()
        proof = _assess_surface_evidence(raw, AS_OF, curves, requests, responses)
        self.assertEqual(proof["status"], "verified")
        self.assertEqual(proof["unit"], "percent")
        self.assertEqual(proof["coverage"], {"total": 2, "requested": 2, "verified": 2, "entire_surface_verified": True})
        normalized = _vendor_surface(raw, AS_OF, curves, verification=proof)
        self.assertEqual(normalized["metadata_basis"], "IPA_typed_fields_verification")
        self.assertTrue(normalized["units_verified_by_response"])
        self.assertFalse(normalized["independent_reference"])
        self.assertEqual([point["iv"] for point in normalized["points"]], [.2, .2])
        self.assertIn("option-contracts-eti", normalized["verification"]["official_sources"][0])
        self.assertEqual(normalized["verification"]["response_sha256"], _evidence_hash(responses))

    def test_decimal_identity_is_verified_without_magnitude_guess(self):
        raw, curves, requests, responses = fixture(decimal=True)
        proof = _assess_surface_evidence(raw, AS_OF, curves, requests, responses)
        self.assertEqual(proof["unit"], "decimal")
        self.assertEqual(proof["coverage"]["verified"], 2)

    def test_partial_query_does_not_verify_other_matrix_cells(self):
        raw, curves, requests, responses = fixture()
        requests[0]["universe"] = requests[0]["universe"][:1]
        responses[0]["data"] = responses[0]["data"][:1]
        proof = _assess_surface_evidence(raw, AS_OF, curves, requests, responses)
        self.assertEqual(proof["status"], "partial")
        self.assertFalse(proof["coverage"]["entire_surface_verified"])
        normalized = _vendor_surface(raw, AS_OF, curves, verification=proof)
        self.assertEqual(len(normalized["points"]), 1)

    def test_typed_model_style_dates_source_and_carry_are_checked(self):
        failures = {
            "PricingModelType": "Bachelier", "ExerciseStyle": "AMER",
            "VolatilityType": "Default", "MarketDataDate": "2026-10-05",
            "ValuationDate": "2026-10-05", "UnderlyingRIC": "SPY.P",
            "UnderlyingPrice": 101, "UnderlyingTimeStamp": "Default",
            "UnderlyingPriceSide": "Mid", "DividendType": "ForecastYield",
            "EndDate": "2027-10-07", "Strike": 101,
            "VolatilityPercent": .2, "MarketValueInDealCcy": 99,
        }
        for field, value in failures.items():
            with self.subTest(field=field):
                raw, curves, requests, responses = fixture()
                update_field(responses, field, value)
                proof = _assess_surface_evidence(raw, AS_OF, curves, requests, responses)
                self.assertEqual(proof["coverage"]["verified"], 1)
                self.assertEqual(len(proof["diagnostics"]), 1)

    def test_stale_spot_discount_or_wrong_axes_rejects_matrix(self):
        for category in ("spot", "discount", "axis"):
            raw, curves, _, _ = fixture()
            if category == "spot":
                raw["data"][0]["underlyingSpot"][0]["priceDate"] = "2026-10-05"
            elif category == "discount":
                raw["data"][0]["discountCurve"]["curveParameters"]["marketDataDate"] = "2026-10-05"
            else:
                raw["request_context"]["y_axis"] = "Moneyness"
            self.assertEqual(_surface_cells(raw, AS_OF), [])

    def test_bound_response_hashes_and_recomputed_flags(self):
        raw, curves, requests, responses = fixture()
        proof = _assess_surface_evidence(raw, AS_OF, curves, requests, responses)
        for category in ("surface", "response", "request"):
            changed, changed_proof = deepcopy(raw), deepcopy(proof)
            if category == "surface":
                changed["data"][0]["surface"][1][1] = 21
            elif category == "response":
                update_field(changed_proof["responses"], "VolatilityPercent", 21)
            else:
                changed_proof["requests"][0]["universe"][0]["instrumentDefinition"]["strike"] = 101
            self.assertIsNone(_vendor_surface(changed, AS_OF, curves, verification=changed_proof))
        proof["points"] = [{"iv": 999}]
        proof["coverage"]["verified"] = 999
        normalized = _vendor_surface(raw, AS_OF, curves, verification=proof)
        self.assertEqual(normalized["verification"]["coverage"]["verified"], 2)
        self.assertEqual(normalized["points"][0]["iv"], .2)

    def test_zero_identity_is_ambiguous(self):
        raw, curves, requests, responses = fixture()
        raw["data"][0]["surface"][1][1] = 0
        update_field(responses, "VolatilityPercent", 0)
        proof = _assess_surface_evidence(raw, AS_OF, curves, requests, responses)
        self.assertEqual(proof["coverage"]["verified"], 1)
        self.assertIn("ambiguous", proof["diagnostics"][0]["code"])

    def test_typed_dataset_is_distinct_from_ambiguous_matrix(self):
        raw, curves, requests, responses = fixture()
        # Typed provider data uses a different IV. It cannot verify the old
        # matrix; it can form its own documented-percent Black dataset.
        for index in range(2):
            requests[0]["universe"][index]["pricingParameters"].update(
                dividendType="HistoricalYield", riskFreeRatePercent=0, dividendYieldPercent=0)
        for name, value in (("DividendType", "HistoricalYield"), ("VolatilityPercent", 30),
                            ("MarketValueInDealCcy", 11.923538474048499)):
            update_field(responses, name, value)
        for field in ("RiskFreeRatePercent", "DividendYieldPercent"):
            responses[0]["headers"].append({"name": field})
            for row in responses[0]["data"]:
                row.append(0)
        update_field(responses, "DividendType", "HistoricalYield", row=1)
        proof = _assess_surface_evidence(raw, AS_OF, curves, requests, responses, typed_dataset=True)
        self.assertEqual(proof["coverage"]["verified"], 2)
        self.assertEqual(proof["unit"], "percent")
        self.assertFalse(proof["matrix_values_verified"])
        self.assertEqual(proof["points"][0]["iv"], .3)
        self.assertEqual(proof["points"][0]["captured_dividend_type"], "ImpliedYield")
        self.assertEqual(proof["points"][0]["dividend_type"], "HistoricalYield")
        normalized = _vendor_surface(raw, AS_OF, curves, verification=proof)
        self.assertEqual(normalized["source"], "LSEG IPA FinancialContracts SVISurface")
        self.assertEqual(normalized["points"][0]["iv"], .3)
        self.assertFalse(normalized["verification"]["matrix_values_verified"])
        update_field(proof["responses"], "DividendYieldPercent", .01)
        proof["response_sha256"] = _evidence_hash(proof["responses"])
        normalized = _vendor_surface(raw, AS_OF, curves, verification=proof)
        self.assertEqual(len(normalized["points"]), 1)

    def test_legacy_assumption_rejected_by_analytics(self):
        from tools.market_validation.analytics import _vendor_surface as accept_surface
        surface = {"as_of": AS_OF.isoformat(), "volatility_unit": "decimal", "convention": "black",
                   "points": [{"expiry": EXPIRY, "strike": 100, "iv": .2}]}
        diagnostics = []
        self.assertIsNotNone(accept_surface({"vendor_surface": surface}, AS_OF, diagnostics))
        for key, value in (("metadata_basis", "SDK_example_convention"), ("unit_assumption", "percent"), ("convention_assumption", "Black")):
            diagnostics = []
            self.assertIsNone(accept_surface({"vendor_surface": {**surface, key: value}}, AS_OF, diagnostics))
            self.assertEqual(diagnostics[-1]["code"], "vendor_surface_unverified")

    def test_explicit_dated_spot_only_accepts_confirmed_user_price(self):
        raw, curves, requests, responses = fixture()
        for index, instrument in enumerate(requests[0]["universe"]):
            instrument["pricingParameters"].update(
                dividendType="HistoricalYield", riskFreeRatePercent=0,
                dividendYieldPercent=0, underlyingPrice=100)
            update_field(responses, "DividendType", "HistoricalYield", row=index)
            update_field(responses, "UnderlyingPriceSide", "User", row=index)
        for field in ("RiskFreeRatePercent", "DividendYieldPercent"):
            responses[0]["headers"].append({"name": field})
            for row in responses[0]["data"]:
                row.append(0)
        proof = _assess_surface_evidence(raw, AS_OF, curves, requests, responses, typed_dataset=True)
        self.assertEqual(proof["coverage"]["verified"], 2)
        self.assertIn("captured", proof["points"][0]["spot_basis"])
        for category in ("request", "returned_side", "returned_price"):
            wanted, returned = deepcopy(requests), deepcopy(responses)
            if category == "request":
                wanted[0]["universe"][0]["pricingParameters"]["underlyingPrice"] = 99
            elif category == "returned_side":
                update_field(returned, "UnderlyingPriceSide", "Last")
            else:
                update_field(returned, "UnderlyingPrice", 99)
            proof = _assess_surface_evidence(raw, AS_OF, curves, wanted, returned, typed_dataset=True)
            self.assertEqual(proof["coverage"]["verified"], 1)

    def test_provider_error_is_retained_before_missing_output_metadata(self):
        raw, curves, requests, responses = fixture()
        update_field(responses, "ErrorCode", "QPS-DPS.1011")
        update_field(responses, "ErrorMessage", "Market data error: access denied to /.SPX")
        update_field(responses, "VolatilityType", None)
        proof = _assess_surface_evidence(raw, AS_OF, curves, requests, responses)
        self.assertIn("provider_point_error", proof["diagnostics"][0]["code"])
        self.assertIn("access denied", proof["diagnostics"][0]["code"])
        # Valid provider rows can spell an enum in uppercase and use a blank
        # error string; neither should hide an otherwise verified response.
        raw, curves, requests, responses = fixture()
        update_field(responses, "ErrorMessage", " ")
        update_field(responses, "CallPut", "CALL")
        proof = _assess_surface_evidence(raw, AS_OF, curves, requests, responses)
        self.assertEqual(proof["coverage"]["verified"], 2)

    def test_capture_failure_survives_offline_validation(self):
        from tools.market_validation.analytics import validate_snapshot
        diagnostics = {"vendor_surface_verification": {"status": "error", "code": "access_denied"}}
        snapshot = {"schema_version": 1, "source": {"as_of": AS_OF.isoformat(), "price_basis": "close"},
                    "spot": 100, "quotes": [], "curves": [], "diagnostics": diagnostics}
        result = validate_snapshot(snapshot)
        self.assertEqual(result["capture_diagnostics"], diagnostics)
        self.assertIsNone(result["vendor_surface"])


if __name__ == "__main__":
    unittest.main()
