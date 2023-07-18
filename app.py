import time
import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from model import open_data,  split_data, load_model_and_predict
from PIL import Image


def write_user_data(df):
    st.write("## Customer's information")
    st.write(df)


def write_prediction(prediction, prediction_probas):
    st.write("## Prediction")

    if prediction == "The customer does not churn!":
        st.success("The customer does not churn!  :white_check_mark:")
    elif prediction == "The customer wants to churn ...":
        st.error('The customer wants to churn... :heavy_exclamation_mark:')
    else:
        st.error('Something went wrong')

    st.write("## Prediction probability")
    st.write(prediction_probas)


def pack_input(contract_type, num_referrals, fiber_optic, premium, unlimited,
               monthly_charge, total_charges, online_security, paperless_billing, credit_card,
               senior, internet_service, married, dependents):
    """ translate input values to pass to model """

    translatetion = {"Yes": 1, "No": 0}

    data = {"Number of Referrals": num_referrals,
            "Monthly Charge": monthly_charge,
            "Total Charges": total_charges,
            "Senior Citizen_Yes": translatetion[senior],
            "Married_Yes": translatetion[married],
            "Dependents_Yes": translatetion[dependents],
            "Internet Service_Yes": translatetion[internet_service],
            "Internet Type_Fiber Optic": translatetion[fiber_optic],
            "Online Security_Yes": translatetion[online_security],
            "Premium Tech Support_Yes": translatetion[premium],
            "Unlimited Data_Yes": translatetion[unlimited],
            "Contract_One Year": 1 if contract_type == 'One year' else 0,
            "Contract_Two Year": 1 if contract_type == 'Two years' else 0,
            "Paperless Billing_Yes": translatetion[paperless_billing],
            "Payment Method_Credit Card": translatetion[credit_card]}

    return pd.DataFrame(data, index=[0])


def render_page():
    st.set_page_config(layout="wide", page_title="Churn Value prediction", page_icon=':runner:')

    churn = Image.open('data/churn.png')
    ratio_churn = Image.open('data/ratio_churn.png')
    tenure = Image.open('data/tenure.png')
    socdem = Image.open('data/socdem.png')
    tenure_cohorts = Image.open('data/tenure_cohorts.png')
    cohortscontract = Image.open('data/cohortscontract.png')
    importances = Image.open('data/importances.png')

    st.title('Churn Value analysis for a telecommunication company')
    st.subheader("Explore customers, "
                 "predict probabilities of churn for a customer with a number of  characteristics,"
                 " evaluate the importance of features.")
    st.write('**Data**: The Telco customer churn data set contains information about a company that provided phone and Internet services  in California.'
             ' It indicates which customers have left, stayed, or signed up for their service.')
    st.image(churn)

    tab1, tab2, tab3 = st.tabs([':mag: Explore', ':mage: Predict', ':vertical_traffic_light: Evaluate'])

    with tab1:
        st.write("### Let's have a look at our customers")
        st.image(socdem)
        st.write("Gender and marital status characteristics are evenly spread.")
        st.write("Most of our customers are in age between 30 and 65.")
        st.write("1/5 of them have dependents (children, parents, grandparents, etc.). ")
        st.divider()

        st.image(ratio_churn)
        st.write("26,5%  of customers churned and 73,5% stayed.")
        st.write("Our variable 'Churn Value' is imbalanced.  It means that we need to use the class weights in models "
                 "as one of possible ways to deal with that problem.")
        st.divider()

        st.image(tenure)
        st.write('Tenure of customers varies from 1 month to 72 months.')
        st.write('In average our customers spend 2,5 years with us (32.39 months).')
        st.divider()

        st.write('There are too many tenure groups. We divide them into 6 cohorts: ')
        st.write('- 0-12 months;')
        st.write('- 13-24 months;')
        st.write('- 25-36 months;')
        st.write('- 37-48 months;')
        st.write('- 49-60 months;')
        st.write('- 61 + months.;')
        st.divider()

        st.image(tenure_cohorts)
        st.write('We see that there are less customers who churned among those who joined us more than 1 year ago. '
                 'And more customers who churned among those who joined 0-12 month ago.')
        st.divider()

        st.image(cohortscontract)
        st.write('We see that customers with a Month to Month contract churn more among all the cohorts. ')
        st.write(' Among customers with One and Two years contract number of those who churn is relatively low. '
        'Most of the customers who decided to churn own the Month to Month contract type.')
        st.write("To discover a Churn prediction model for customers with "
                 " [Month to Month contract type, click here](https://churn-prediction-telco-monthly.streamlit.app)")
        st.divider()

    with tab2:
        st.write("""### Enter information about your customer""")

        col1, col2 = st.columns(2)
        with col1:
            contract_type = st.selectbox("Which type of the contract has the customer?", ('Month to Month ', "One year", "Two years"))

        with col2:
            num_referrals = st.slider("How many referrals did the customer", min_value=0, max_value=11, value=0, step=1)

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            monthly_charge = st.slider("Choose a Monthly Charge of a customer", min_value=15, max_value=120, value=20, step=5)

        with col2:
            total_charges = st.slider("Choose Total Charges of a customer", min_value=15, max_value=8000, value=20, step=5)
        st.divider()

        st.write('**Please, choose "Yes" or "No"**')
        col1, col2 = st.columns(2)
        with col1:
            senior = st.selectbox("Is the customer 65 or older?", ("Yes", "No"))
            married = st.selectbox("Is the customer married?", ("Yes", "No"))
            dependents = st.selectbox("Does the customer live with any dependents? (children, parents, grandparents, etc.)", ("Yes", "No"))
            paperless_billing = st.selectbox("Does the customer use paperless billing?", ("Yes", "No"))
            credit_card = st.selectbox("Does the customer use a credit card as a payment method?", ("Yes", "No"))

        with col2:
            internet_service = st.selectbox("Does the customer use Internet services?", ("Yes", "No"))
            unlimited = st.selectbox("Does the customer use unlimited data?", ("Yes", "No"))
            fiber_optic = st.selectbox("Does the customer use fiber optic?", ("Yes", "No"))
            premium = st.selectbox("Does the customer use premium tech support?", ("Yes", "No"))
            online_security = st.selectbox("Does the customer use Online Security service?", ("Yes", "No"))

        col1, col2, col3 = st.columns(3)
        if col2.button('To predict'):
            with st.spinner('Evaluating!'):
                time.sleep(1)
                user_input_df = pack_input(contract_type, num_referrals, fiber_optic, premium, unlimited,
                                    monthly_charge, total_charges, online_security, paperless_billing, credit_card,
                                    senior, internet_service, married, dependents)

                train_df = open_data()
                train_X_df, _ = split_data(train_df)
                full_X_df = pd.concat((user_input_df, train_X_df), axis=0)

                ss = MinMaxScaler()
                ss.fit(full_X_df)

                full_X_df = pd.DataFrame(ss.transform(full_X_df), columns=full_X_df.columns)

                user_X_df = full_X_df[:1]
                write_user_data(user_X_df)

                prediction, prediction_probas = load_model_and_predict(user_X_df)
                write_prediction(prediction, prediction_probas)


    with tab3:
        st.write('### What are important features for the Churn Value classification?')
        st.image(importances)


if __name__ == "__main__":
    render_page()