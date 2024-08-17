import streamlit as st
import pandas as pd

# Function to check if user already exists
def user_exists(email):
    try:
        df = pd.read_csv('C:/Users/Asus/PycharmProjects/pythonProject/sign.csv')
        return email in df['Email'].values
    except FileNotFoundError:
        return False

# Function to register new user
def register_user(email, password):
    try:
        df = pd.read_csv('C:/Users/Asus/PycharmProjects/pythonProject/sign.csv')
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Email', 'Password'])
    new_data = pd.DataFrame({'Email': [email], 'Password': [password]})
    df = pd.concat([df, new_data], ignore_index=True)

    df.to_csv('C:/Users/Asus/PycharmProjects/pythonProject/sign.csv', index=False)


# Function to authenticate user
def authenticate_user(email, password):
    try:
        df = pd.read_csv('C:/Users/Asus/PycharmProjects/pythonProject/sign.csv')
        return (email, password) in zip(df['Email'], df['Password'])
    except FileNotFoundError:
        return False

# Register Page
def register_page():
    st.title('Register')
    email = st.text_input('Email')
    password = st.text_input('Password', type='password')
    if st.button('Register'):
        if not email or not password:
            st.error('Please enter both email and password.')
        elif user_exists(email):
            st.error('User with this email already exists.')
        else:
            register_user(email, password)
            st.success('Registration successful!')
            st.switch_page('naive.py')

# Login Page
def login_page():
    st.title('Login')
    email = st.text_input('Email')
    password = st.text_input('Password', type='password')
    if st.button('Login'):
        if not email or not password:
            st.error('Please enter both email and password.')
        elif authenticate_user(email, password):
            st.success('Login successful!')
            st.write('Welcome to the dashboard!')
        else:
            st.error('Invalid email or password.')

def main():
    page = st.sidebar.radio("Navigation", ["Login", "Register"])
    if page == 'Login':
        login_page()
    elif page == 'Register':
        register_page()

if __name__ == '__main__':
    main()
