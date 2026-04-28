def get_retention_recommendation(customer_row):
    recommendations = []

    monthly_charges = float(customer_row.get("MonthlyCharges", 0) or 0)
    tenure = int(customer_row.get("tenure", 0) or 0)
    contract = str(customer_row.get("Contract", "") or "")
    tech_support = str(customer_row.get("TechSupport", "") or "")
    internet_service = str(customer_row.get("InternetService", "") or "")
    payment_method = str(customer_row.get("PaymentMethod", "") or "")
    senior_citizen = int(customer_row.get("SeniorCitizen", 0) or 0)
    paperless_billing = str(customer_row.get("PaperlessBilling", "") or "")
    online_security = str(customer_row.get("OnlineSecurity", "") or "")
    has_multiple_services = customer_row.get("HasMultipleServices", 0)
    
    # Ensure has_multiple_services is an integer
    try:
        has_multiple_services = int(has_multiple_services)
    except (ValueError, TypeError):
        has_multiple_services = 0

    # Rule 1: High monthly charges
    if monthly_charges > 70:
        recommendations.append("💰 Offer a 15% discounted monthly plan")

    # Rule 2: New customer with low tenure
    if tenure < 12:
        recommendations.append("🤝 Assign a dedicated customer success manager")

    # Rule 3: Month-to-month contract
    if contract == "Month-to-month":
        recommendations.append("📋 Offer incentives to upgrade to a 1-year or 2-year contract")

    # Rule 4: No tech support
    if tech_support == "No":
        recommendations.append("🔧 Offer free TechSupport upgrade for 3 months")

    # Rule 5: Fiber optic with high charges
    if internet_service == "Fiber optic" and monthly_charges > 80:
        recommendations.append("🌐 Offer a Fiber Optic loyalty discount")

    # Rule 6: Electronic check payment (highest churn indicator)
    if "electronic check" in payment_method.lower():
        recommendations.append("🏦 Offer 5% discount for switching to auto-pay (credit card or bank transfer)")

    # Rule 7: Senior citizen
    if senior_citizen == 1:
        recommendations.append("👴 Enroll in Senior Loyalty Program with discounted rates")

    # Rule 8: No online security
    if online_security == "No" and internet_service != "No":
        recommendations.append("🔒 Offer free Online Security add-on for 2 months")

    # Rule 9: Low service adoption
    if has_multiple_services < 2:
        recommendations.append("📦 Suggest a bundled services package for better value")

    # Rule 10: Paperless billing with high charges
    if paperless_billing == "Yes" and monthly_charges > 65:
        recommendations.append("📧 Send exclusive paperless customer loyalty reward")

    # Rule 11: Long-tenure customer at risk (valuable customer)
    if tenure > 24 and monthly_charges > 60:
        recommendations.append("⭐ Offer VIP loyalty reward for being a valued long-term customer")

    # Always include baseline
    recommendations.append("📩 Send a personalized retention email with a special offer")

    return recommendations
