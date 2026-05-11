import unittest

from rv_heston_hmm.kalshi import KalshiClient


class FakeKalshiClient(KalshiClient):
    def __init__(self, payloads):
        super().__init__(base_url="https://example.invalid")
        self.payloads = payloads

    def _get(self, path, params=None):
        return self.payloads[path]


class KalshiParsingTests(unittest.TestCase):
    def test_orderbook_derived_yes_ask_from_no_bid(self):
        client = FakeKalshiClient(
            {
                "/markets/TEST/orderbook": {
                    "orderbook": {
                        "yes": [[40, 10]],
                        "no": [[55, 12]],
                    }
                }
            }
        )
        top = client.get_book_top("TEST")
        self.assertEqual(top.yes_bid, 0.40)
        self.assertEqual(top.yes_bid_qty, 10.0)
        self.assertAlmostEqual(top.yes_ask, 0.45)
        self.assertEqual(top.yes_ask_qty, 12.0)

    def test_fallback_to_market_top_of_book(self):
        client = FakeKalshiClient(
            {
                "/markets/TEST/orderbook": {"orderbook": {}},
                "/markets/TEST": {
                    "market": {
                        "yes_bid_dollars": "0.2300",
                        "yes_ask_dollars": "0.2700",
                        "yes_bid_size_fp": "100.00",
                        "yes_ask_size_fp": "200.00",
                    }
                },
            }
        )
        top = client.get_book_top("TEST")
        self.assertEqual(top.yes_bid, 0.23)
        self.assertEqual(top.yes_ask, 0.27)
        self.assertEqual(top.yes_bid_qty, 100.0)
        self.assertEqual(top.yes_ask_qty, 200.0)


if __name__ == "__main__":
    unittest.main()

