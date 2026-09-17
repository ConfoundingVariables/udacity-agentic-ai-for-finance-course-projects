"""Unit tests for the evaluator-optimizer validation logic (offline)."""

from agents.evaluator_optimizer import EvaluatorOptimizerPattern


def _pattern():
    return EvaluatorOptimizerPattern()


def test_valid_message_passes():
    msg = {
        'message_id': 'VALID001',
        'message_type': 'MT103',
        'reference': 'REF123456',
        'amount': '5000.00 USD',
        'sender_bic': 'CHASUS33XXX',
        'receiver_bic': 'DEUTDEFFXXX',
        'remittance_info': 'Invoice payment',
    }
    is_valid, errors = _pattern().evaluate_message(msg)
    assert is_valid is True
    assert errors == []


def test_invalid_message_collects_errors():
    msg = {
        'message_id': 'INVALID001',
        'message_type': 'MT999',                       # invalid type
        'reference': 'THIS_REFERENCE_IS_WAY_TOO_LONG_12345',  # too long
        'amount': '9999999999.99 USD',                 # exceeds max
        'sender_bic': 'INVALID',                       # bad BIC
        'receiver_bic': 'INVALID',                     # bad BIC + same as sender
        'remittance_info': 'Test',
    }
    is_valid, errors = _pattern().evaluate_message(msg)
    assert is_valid is False
    joined = " ".join(errors)
    assert "Invalid message type" in joined
    assert "Reference too long" in joined
    assert "Amount exceeds maximum" in joined
    assert "Invalid sender BIC" in joined
    assert "Sender and receiver BIC cannot be the same" in joined


def test_generator_format_with_separate_currency_field():
    """The generator emits a bare numeric amount + a separate currency field."""
    msg = {
        'message_id': 'GEN001',
        'message_type': 'MT103',
        'reference': 'PAY123456',
        'amount': '6938.06',          # no embedded currency
        'currency': 'CHF',            # dedicated field
        'sender_bic': 'NSVQCHADT29',
        'receiver_bic': 'CERQSG3JA76',
    }
    is_valid, errors = _pattern().evaluate_message(msg)
    assert is_valid is True, errors


def test_unsupported_currency_flagged():
    msg = {
        'message_id': 'CUR001',
        'message_type': 'MT103',
        'reference': 'PAY1',
        'amount': '100.00',
        'currency': 'XYZ',            # not a recognised currency
        'sender_bic': 'CHASUS33XXX',
        'receiver_bic': 'DEUTDEFFXXX',
    }
    is_valid, errors = _pattern().evaluate_message(msg)
    assert is_valid is False
    assert any("Invalid currency: XYZ" in e for e in errors)


def test_missing_required_fields():
    is_valid, errors = _pattern().evaluate_message({'message_id': 'X'})
    assert is_valid is False
    assert any("Missing required field" in e for e in errors)


def test_validate_bic_variants():
    p = _pattern()
    assert p._validate_bic("CHASUS33XXX") is True   # 11 chars
    assert p._validate_bic("CHASUS33") is True       # 8 chars
    assert p._validate_bic("INVALID") is False       # 7 chars
    assert p._validate_bic("TEST1234") is False      # digits in country code
    assert p._validate_bic("") is False


def test_optimize_returns_message_when_no_errors():
    msg = {'message_id': 'X', 'amount': '10.00 USD'}
    assert _pattern().optimize_message(msg, []) is msg


def test_optimize_offline_preserves_required_fields():
    """Offline, the correction agent returns the original message unchanged."""
    msg = {
        'message_id': 'FIX001',
        'message_type': 'MT103',
        'reference': 'REF1',
        'amount': '100.00 USD',
        'sender_bic': 'CHASUS33XXX',
        'receiver_bic': 'DEUTDEFFXXX',
    }
    corrected = _pattern().optimize_message(msg, ["Some error"])
    for field in ("message_type", "reference", "amount", "sender_bic", "receiver_bic"):
        assert field in corrected
