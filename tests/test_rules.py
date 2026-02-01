from detectors.compliance_detector import ComplianceDetector

def test_rules_exposed_hair():
    det = ComplianceDetector(model_path="models/best.onnx")
    ok, violations, reason = det._apply_rules(["exposed_hair"])
    assert ok is False
    assert "exposed_hair" in violations

def test_rules_head_net_ok():
    det = ComplianceDetector(model_path="models/best.onnx")
    ok, violations, reason = det._apply_rules(["head_net"])
    assert ok is True

def test_rules_missing_head_net_if_beard_only():
    det = ComplianceDetector(model_path="models/best.onnx")
    ok, violations, reason = det._apply_rules(["beard_net"])
    assert ok is False
    assert "missing_head_net" in violations
