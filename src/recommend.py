def get_retention_recommendation(customer_row):
    """
    Takes one customer's data as a dictionary (original values, not encoded).
    Returns a list of recommendation strings.
    """
    recommendations = []

    monthly_charges = customer_row.get("MonthlyCharges", 0)
    tenure = customer_row.get("tenure", 0)
    contract = customer_row.get("Contract", "")
    tech_support = customer_row.get("TechSupport", "")
    internet_service = customer_row.get("InternetService", "")

    if monthly_charges > 70:
        recommendations.append("Offer a discounted monthly plan")

    if tenure < 12:
        recommendations.append("Assign a dedicated customer success manager")

    if contract == "Month-to-month":
        recommendations.append("Offer a 1-year or 2-year contract discount")

    if tech_support == "No":
        recommendations.append("Offer free tech support upgrade for 3 months")

    if internet_service == "Fiber optic" and monthly_charges > 80:
        recommendations.append("Offer fiber loyalty discount")

    # Always include this as a baseline recommendation
    recommendations.append("Send a personalized retention email")

    return recommendations


if __name__ == "__main__":
    test_customer = {
        "MonthlyCharges": 85,
        "tenure": 6,
        "Contract": "Month-to-month",
        "TechSupport": "No",
        "InternetService": "Fiber optic"
    }

    print("Test Customer:")
    for key, value in test_customer.items():
        print(f"  {key}: {value}")

    print()
    print("Recommendations:")
    recommendations = get_retention_recommendation(test_customer)
    for rec in recommendations:
        print(f"  - {rec}")

    print()
    print(f"Total recommendations: {len(recommendations)}")
 
