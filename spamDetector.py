import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open('spam.pkl', 'rb'))
cv = pickle.load(open('vectorizer.pkl', 'rb'))

def main():
    st.title("Email Spam Classification Application")
    st.write("This is a Machine Learning application to classify emails as spam or ham.")
    st.subheader("Classification")
    
    user_input = st.text_area("Enter an email to classify", height=150)
    if st.button("Classify"):
        if user_input:
            data = [user_input]  # Collect user input
            st.write("Processing your input...")  # Optional debugging message
            
            vec = cv.transform(data).toarray()  # Vectorize the input
            result = model.predict(vec)  # Predict spam or not spam
            
            if result[0] == 0:
                st.success("This is Not A Spam Email")
            else:
                st.error("This is A Spam Email")
        else:
            st.write("Please enter an email to classify.")  # Prompt user to enter text

# Run the main function
main()
