from .message_schemas import OverrideLevel


def required_autonomy_level(jensen_gain_deg: float,
                             physics_consistent: bool,
                             is_in_distribution: bool,
                             novelty_score: float) -> dict:
    severity_score = 0.0
    reasons = []

    if jensen_gain_deg > 20.0:
        severity_score += 2; reasons.append("high pose uncertainty (Jensen Gain)")
    elif jensen_gain_deg > 10.0:
        severity_score += 1; reasons.append("moderate pose uncertainty (Jensen Gain)")

    if not physics_consistent:
        severity_score += 2; reasons.append("physics cross-check failed")

    if not is_in_distribution:
        severity_score += 3; reasons.append("out-of-distribution input")

    if novelty_score > 0.7:
        severity_score += 2; reasons.append("novel/unrecognized situation")
    elif novelty_score > 0.4:
        severity_score += 1; reasons.append("moderately novel situation")

    if severity_score == 0:
        level = "AUTONOMOUS"
    elif severity_score <= 2:
        level = OverrideLevel.ACKNOWLEDGE
    elif severity_score <= 4:
        level = OverrideLevel.MODIFY
    else:
        level = OverrideLevel.REPLACE

    return {"required_level": level, "severity_score": severity_score, "reasons": reasons}
