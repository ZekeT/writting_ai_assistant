from frontend.ui_components import (
    render_center_column,
    render_right_column,
    render_copy_modal,
    render_sidebar_content
)
from backend.session_manager import SessionManager
from backend.database import get_db_instance
import streamlit as st
import sys
import os
from datetime import datetime

# Add the parent directory to the path to import backend modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    st.set_page_config(
        page_title="Writing Assistant",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    SessionManager.initialize_session()

    # Initialize database connection
    db = get_db_instance()

    # Sidebar for publication selection and left panel content
    with st.sidebar:
        st.title("Writing Assistant")

        # Get publications from database
        publications = db.get_publications()

        # Create dropdown for publication selection
        publication_options = [pub['name'] for pub in publications]
        selected_publication = st.selectbox(
            "Select Publication", publication_options)

        # Update session state when publication changes
        selected_pub_id = next(
            (pub['id'] for pub in publications if pub['name'] == selected_publication), None)

        if (not st.session_state.current_publication or
                st.session_state.current_publication['id'] != selected_pub_id):
            SessionManager.set_current_publication(
                selected_pub_id, selected_publication)
            # Reset current draft when publication changes
            SessionManager.set_current_draft(None)

        # Render former left panel content in the sidebar
        render_sidebar_content()

    # Main content area with two columns
    center_col, right_col = st.columns([2, 1])

    with center_col:
        render_center_column()

    with right_col:
        render_right_column()


if __name__ == "__main__":
    main()
