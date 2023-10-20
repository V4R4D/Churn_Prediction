import streamlit as st
import pandas as pd
import subprocess

# Check if required dependencies are installed, and if not, install them
def install_dependencies():
    try:
        import joblib
        import pandas
        import sklearn
        
    except ImportError as e:
        # If any of the required modules are missing, install them
        subprocess.run(["pip", "install", "-r", "requirements.txt"])

# Call the function to install dependencies
install_dependencies()



from sklearn.preprocessing import StandardScaler , LabelEncoder

# Load the model here
import joblib
model = joblib.load(r"Model/model.sav")
# model = load_model()

# sample dataframe created with required columns to fit in standard scalar...
df_sample = pd.DataFrame(columns=['Age', 'Gender', 'Location', 'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB']) #Sample dataframe 

df_sample.loc[0] = [63, 'Male', 'Los Angeles', 17, 73.36, 236] # sample column to fit for StandardScalar....

gender_mapping = {'Male': 1, 'Female': 0}
location_mapping = {'Los Angeles': 0, 'Chicago': 1, 'Miami': 2, 'New York': 3, 'Houston': 4}

df_sample['Gender'] = df_sample['Gender'].map(gender_mapping)
df_sample['Location'] = df_sample['Location'].map(location_mapping)


def main():
    

    st.markdown("<h3></h3>", unsafe_allow_html=True)

    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?", ("Online", "Batch"))
    st.sidebar.info('This app is created to predict Customer Churn')

    if add_selectbox == "Online":
        st.info("Input data below")

        # Create a StandardScaler for feature scaling
        sc = StandardScaler()

        st.title('Customer Churn Prediction Model')

        # User input section
        st.subheader("Demographic data")


        # User input section
        Gender = st.selectbox('Gender:', ('Male', 'Female'))
        Age = st.number_input('The age of the customer', min_value=0, max_value=100, value=0)
        Location = st.selectbox('Location:', ('Los Angeles', 'Chicago', 'Miami', 'New York', 'Houston'))
        
        df_sample['Gender'] = df_sample['Gender'].map(gender_mapping)

# Apply location mapping to the 'Location' column
        df_sample['Location'] = df_sample['Location'].map(location_mapping)

        # Convert Gender and Location to integers using the mapping dictionaries
        gender_numeric = gender_mapping[Gender]
        location_numeric = location_mapping[Location]

        st.subheader("Your Data Plan")
        Subscription_Length_Months = st.slider('Number of months the customer has stayed with the company', min_value=0, max_value=72, value=0)
        Monthly_Bill = st.number_input('The amount charged to the customer monthly', min_value=0, max_value=150000, value=0)
        Total_Usage_GB = st.number_input('The total usage in GB of the customer', min_value=0, max_value=1000, value=0)



        # Prepare user input data as a dictionary in correct order
        user_inputs = {
            'Age': Age,
            'Gender': gender_numeric,  
            'Location': location_numeric, 
            'Subscription_Length_Months': Subscription_Length_Months,
            'Monthly_Bill': Monthly_Bill,
            'Total_Usage_GB': Total_Usage_GB
        }

        
        # Select the columns to be used for scaling
        selected_columns = ['Age', 'Gender', 'Location', 'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB']

        # Fit the scaler with the selected columns of your training data
        sc.fit(df_sample[selected_columns])

        preprocess_df = pd.DataFrame.from_dict([user_inputs])
        

        if st.button('Predict'):
            prediction = model.predict(preprocess_df)
            if prediction == 1:
                st.warning('Yes, the customer will terminate the service.')
            else:
                st.success('No, the customer is happy with Telco Services.')

    else:
        st.subheader("Dataset upload")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)

            st.write(data.head())
            st.markdown("<h3></h3>", unsafe_allow_html=True)



            if st.button('Predict'):
                # Use your model to predict
                prediction = model.predict(data)

                prediction_df = pd.DataFrame(prediction, columns=["Predictions"])
                prediction_df = prediction_df.replace({1: 'Yes, the customer will terminate the service.', 0: 'No, the customer is happy with Telco Services.'})

                st.markdown("<h3></h3>", unsafe_allow_html=True)
                st.subheader('Prediction')
                st.write(prediction_df)

if __name__ == '__main__':
    main()
