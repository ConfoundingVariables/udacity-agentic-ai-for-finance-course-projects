"""Unit tests for the parallel fraud-screening pattern (offline)."""

from agents.parallelization import ParallelizationPattern


def test_pattern_has_four_agents_and_aggregator():
    p = ParallelizationPattern()
    assert len(p.list_of_agents) == 4
    assert p.aggregator is not None


def test_clean_message_marked_clean():
    p = ParallelizationPattern()
    msg = {
        'message_id': 'CLEAN01',
        'amount': '1200.00 USD',
        'sender_bic': 'CHASUS33XXX',
        'receiver_bic': 'DEUTDEFFXXX',
        'remittance_info': 'Invoice payment',
    }
    out = p.process_batch_parallel([msg])[0]
    assert out['fraud_status'] == 'CLEAN'
    assert 'fraud_analysis' in out
    assert len(out['fraud_analysis']) == 4


def test_obviously_fraudulent_message_flagged():
    p = ParallelizationPattern()
    msg = {
        'message_id': 'FRAUD01',
        'amount': '9000000.00 USD',
        'sender_bic': 'TESTRU33XXX',
        'receiver_bic': 'TESTRU33XXX',
        'remittance_info': 'urgent secret transfer',
    }
    out = p.process_batch_parallel([msg])[0]
    assert out['fraud_status'] == 'FRAUDULENT'
    assert out['fraud_score'] > 0
    assert out['fraud_reasons']
