"""Customer email templates. Each template carries structured metadata
(`meta`) that the grader uses to compute the ground-truth correct action
under the current policy. The agent does NOT see `meta`.

Refund amounts are chosen to span all policy caps ($25/$50/$100/$200) so
every drift scenario produces boundary-flipping cases.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CustomerTemplate:
    id: str
    subject: str
    body: str
    kind: str
    refund_amount: float | None = None
    severity: str | None = None
    needs_info: str | None = None


CUSTOMER_TEMPLATES: list[CustomerTemplate] = [
    # ---------- refunds at varied amounts, crossing all caps ----------
    CustomerTemplate(
        "refund_15", "Small refund for cancelled add-on",
        "Hi, please refund the $15 add-on I cancelled yesterday. Account 18244.",
        "refund", refund_amount=15.0,
    ),
    CustomerTemplate(
        "refund_30", "Partial month refund",
        "I downgraded mid-cycle on April 8 and was overcharged $30. Please refund.",
        "refund", refund_amount=30.0,
    ),
    CustomerTemplate(
        "refund_40", "Charged twice for my subscription",
        "Hi, my card was charged $40 twice this month. Order IDs 82811 and 82812. "
        "Please refund the duplicate.",
        "refund", refund_amount=40.0,
    ),
    CustomerTemplate(
        "refund_50", "Refund for unused trial upgrade",
        "I upgraded but never used the premium features — please refund the $50.",
        "refund", refund_amount=50.0,
    ),
    CustomerTemplate(
        "refund_60", "Service interruption refund",
        "We lost ~3 hours of service last night. The pro-rated refund should be $60.",
        "refund", refund_amount=60.0,
    ),
    CustomerTemplate(
        "refund_75", "Refund for bad experience",
        "Your service was down during our event. We paid $75 for the premium "
        "slot and couldn't use it. Please refund the $75.",
        "refund", refund_amount=75.0,
    ),
    CustomerTemplate(
        "refund_90", "Misquoted setup fee",
        "I was quoted a $90 setup fee that I was told would be waived, and it "
        "was still charged. Please refund $90.",
        "refund", refund_amount=90.0,
    ),
    CustomerTemplate(
        "refund_100", "Failed project kickoff refund",
        "The onboarding session never happened. Please refund the $100 fee.",
        "refund", refund_amount=100.0,
    ),
    CustomerTemplate(
        "refund_120", "Duplicate annual renewal",
        "I was charged $120 for a plan I already renewed through my partner "
        "account. Please reverse one of the two charges.",
        "refund", refund_amount=120.0,
    ),
    CustomerTemplate(
        "refund_150", "Refund for cancelled annual plan",
        "I cancelled the annual plan within the 30-day window but still got "
        "charged $150. Please reverse it.",
        "refund", refund_amount=150.0,
    ),
    CustomerTemplate(
        "refund_180", "Misapplied promo code",
        "The BLACKFRIDAY code I used never applied. I overpaid by $180.",
        "refund", refund_amount=180.0,
    ),
    CustomerTemplate(
        "refund_250", "Wrong tier charged",
        "I'm on the Pro plan but was billed for Enterprise ($250 delta). "
        "Please refund the difference.",
        "refund", refund_amount=250.0,
    ),

    # ---------- critical incidents ----------
    CustomerTemplate(
        "critical_outage", "URGENT: Production is down",
        "Our entire production environment is offline since 8 AM. This is "
        "critical — customers can't check out. Please escalate immediately.",
        "critical_incident", severity="critical",
    ),
    CustomerTemplate(
        "critical_data_loss", "Data loss — need urgent help",
        "I just noticed 3 months of customer records are missing from our "
        "account. This is critical for our business.",
        "critical_incident", severity="critical",
    ),
    CustomerTemplate(
        "critical_security", "Possible unauthorised access",
        "I see logins from IPs in two countries I've never been in. This "
        "looks like a security breach — need urgent help.",
        "critical_incident", severity="critical",
    ),
    CustomerTemplate(
        "critical_api_down", "API returning 500 across the board",
        "Every endpoint is returning 500 since ~10 minutes ago. We have a "
        "customer demo in 30 minutes. This is critical.",
        "critical_incident", severity="critical",
    ),
    CustomerTemplate(
        "critical_payment_fail", "Payment processor is rejecting all charges",
        "Stripe integration just started rejecting every transaction. We're "
        "losing live sales. Critical priority.",
        "critical_incident", severity="critical",
    ),
    CustomerTemplate(
        "critical_auth_locked", "All admins locked out",
        "Nobody on our team can sign into the admin panel since the update. "
        "We can't run the business — this is critical.",
        "critical_incident", severity="critical",
    ),

    # ---------- info requests (missing fields) ----------
    CustomerTemplate(
        "info_missing_order_id", "Question about my invoice",
        "Hi, I got an invoice but I don't recognise the line item. Could you "
        "check what it's for?",
        "info_request", needs_info="order_id",
    ),
    CustomerTemplate(
        "info_missing_account", "Help with login",
        "I can't log in anymore. Could you help me get back into my account?",
        "info_request", needs_info="account_email",
    ),
    CustomerTemplate(
        "info_missing_date", "Billing concern",
        "I think there was an extra charge recently but I'm not sure when. "
        "Can you check?",
        "info_request", needs_info="charge_date",
    ),

    # ---------- billing / product questions ----------
    CustomerTemplate(
        "billing_tiers", "How do subscription tiers work?",
        "Quick question — what's the difference between Pro and Team tiers? "
        "Are there usage limits on Team?",
        "billing_q",
    ),
    CustomerTemplate(
        "billing_prorated", "Does annual billing prorate?",
        "If I upgrade mid-cycle on the annual plan, is the delta prorated or "
        "charged in full?",
        "billing_q",
    ),
    CustomerTemplate(
        "billing_invoice_download", "Where do I get past invoices?",
        "Where can I download invoices from prior months? The billing page "
        "only shows the current one.",
        "billing_q",
    ),

    # ---------- chitchat / no action needed ----------
    CustomerTemplate(
        "chitchat_thanks", "Thanks for the quick help",
        "Just wanted to say thanks — the team resolved my issue from last "
        "week really fast. Appreciate it!",
        "chitchat",
    ),
    CustomerTemplate(
        "chitchat_feedback", "Love the new dashboard",
        "Wanted to say the new dashboard looks great. Much cleaner than the "
        "old one. Keep it up!",
        "chitchat",
    ),
    CustomerTemplate(
        "chitchat_ooo", "Out of office until Monday",
        "Hey team, I'll be out until Monday. No action needed — just flagging "
        "so you're not waiting on me for the ticket from last week.",
        "chitchat",
    ),
]
