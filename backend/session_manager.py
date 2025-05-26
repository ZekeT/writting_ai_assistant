from backend.database import get_db_instance
import streamlit as st
import sys
import os
from datetime import datetime

# Add the parent directory to the path to import backend modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SessionManager:
    """
    Manages user session state in Streamlit
    """

    @staticmethod
    def initialize_session():
        """Initialize session state variables if they don't exist"""
        if 'user_id' not in st.session_state:
            st.session_state.user_id = None  # For future login functionality

        if 'username' not in st.session_state:
            st.session_state.username = "Guest"  # Default username

        if 'is_authenticated' not in st.session_state:
            st.session_state.is_authenticated = False

        if 'current_publication' not in st.session_state:
            st.session_state.current_publication = None

        if 'current_draft_id' not in st.session_state:
            st.session_state.current_draft_id = None

        if 'current_draft_date' not in st.session_state:
            st.session_state.current_draft_date = datetime.now().strftime('%Y-%m-%d')

        if 'show_settings' not in st.session_state:
            st.session_state.show_settings = False

        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

    @staticmethod
    def set_current_publication(publication_id, publication_name):
        """Set the current publication"""
        st.session_state.current_publication = {
            'id': publication_id,
            'name': publication_name
        }

    @staticmethod
    def set_current_draft(draft_id):
        """Set the current draft"""
        st.session_state.current_draft_id = draft_id

    @staticmethod
    def set_draft_date(date_str):
        """Set the current draft date"""
        st.session_state.current_draft_date = date_str

    @staticmethod
    def toggle_settings():
        """Toggle settings panel visibility"""
        st.session_state.show_settings = not st.session_state.show_settings

    @staticmethod
    def add_chat_message(role, content):
        """Add a message to the chat history"""
        st.session_state.chat_history.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })

    @staticmethod
    def clear_chat_history():
        """Clear the chat history"""
        st.session_state.chat_history = []

    @staticmethod
    def login_user(user_id, username):
        """Set user as logged in"""
        st.session_state.user_id = user_id
        st.session_state.username = username
        st.session_state.is_authenticated = True

    @staticmethod
    def logout_user():
        """Log out the current user"""
        st.session_state.user_id = None
        st.session_state.username = "Guest"
        st.session_state.is_authenticated = False
