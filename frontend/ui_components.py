from frontend.draft_operations import (
    get_drafts_for_current_publication,
    get_current_draft_content,
    save_draft_content,
    save_current_draft,
    approve_current_draft,
    create_new_draft,
    rename_draft,
    delete_draft,
    copy_draft,
    get_draft_title,
    process_ai_chat
)
from backend.session_manager import SessionManager
from backend.database import get_db_instance
import streamlit as st
import sys
import os
from datetime import datetime

# Add the parent directory to the path to import backend modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def render_sidebar_content():
    """Render the sidebar content (former left column)"""
    # Settings icon
    if st.button("⚙️ Settings", key="settings_button"):
        SessionManager.toggle_settings()

    # Render settings panel directly under the settings button
    if st.session_state.show_settings:
        with st.container():
            st.subheader("Writing Assistant Settings")

            # Example settings
            st.selectbox(
                "AI Model", ["GPT-4", "GPT-3.5", "Claude"], key="ai_model_setting")
            st.slider("Temperature", 0.0, 1.0, 0.7,
                      0.1, key="temperature_setting")
            st.number_input("Max Tokens", 100, 4000, 1000,
                            100, key="max_tokens_setting")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Save", key="save_settings_button"):
                    # In a real implementation, this would save settings to the database
                    st.success("Settings saved!")
                    SessionManager.toggle_settings()
                    st.rerun()

            with col2:
                if st.button("Cancel", key="cancel_settings_button"):
                    SessionManager.toggle_settings()
                    st.rerun()

            st.divider()

    # Date picker for filtering drafts
    selected_date = st.date_input(
        "Select Date",
        value=datetime.strptime(
            st.session_state.current_draft_date, '%Y-%m-%d'),
        key="date_picker"
    )

    # Update session state when date changes
    if selected_date.strftime('%Y-%m-%d') != st.session_state.current_draft_date:
        SessionManager.set_draft_date(selected_date.strftime('%Y-%m-%d'))
        st.rerun()

    # Add new draft button
    if st.button("➕ Add New Draft", key="add_draft_button"):
        st.session_state.show_add_draft_modal = True

    # Render add draft panel directly under the add button
    if hasattr(st.session_state, 'show_add_draft_modal') and st.session_state.show_add_draft_modal:
        with st.container():
            st.subheader("Create New Draft")

            draft_title = st.text_input("Draft Title", key="new_draft_title")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Create", key="create_draft_button") and draft_title:
                    create_new_draft(draft_title)
                    st.session_state.show_add_draft_modal = False
                    st.rerun()

            with col2:
                if st.button("Cancel", key="cancel_add_draft_button"):
                    st.session_state.show_add_draft_modal = False
                    st.rerun()

            st.divider()

    # Draft list
    st.subheader("Draft List")

    # Get drafts from database
    drafts = get_drafts_for_current_publication()

    if not drafts:
        st.info("No drafts found for this date and publication.")
    else:
        for draft in drafts:
            draft_id = draft['id']

            # Draft row with buttons
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                if st.button(f"{draft['title']}", key=f"draft_{draft_id}"):
                    SessionManager.set_current_draft(draft_id)
                    st.rerun()
            with col2:
                if st.button("✏️", key=f"rename_{draft_id}"):
                    # Toggle rename modal for this specific draft
                    if hasattr(st.session_state, 'draft_to_rename') and st.session_state.draft_to_rename == draft_id:
                        st.session_state.draft_to_rename = None
                    else:
                        st.session_state.draft_to_rename = draft_id
            with col3:
                if st.button("🗑️", key=f"delete_{draft_id}"):
                    # Toggle delete modal for this specific draft
                    if hasattr(st.session_state, 'draft_to_delete') and st.session_state.draft_to_delete == draft_id:
                        st.session_state.draft_to_delete = None
                    else:
                        st.session_state.draft_to_delete = draft_id

            # Render rename modal directly under this draft if it's selected
            if hasattr(st.session_state, 'draft_to_rename') and st.session_state.draft_to_rename == draft_id:
                with st.container():
                    st.subheader("Rename Draft")
                    current_title = get_draft_title(draft_id)
                    new_title = st.text_input(
                        "New Title", value=current_title, key=f"rename_title_{draft_id}")

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Save", key=f"save_rename_{draft_id}") and new_title:
                            rename_draft(draft_id, new_title)
                            st.session_state.draft_to_rename = None
                            st.rerun()

                    with col2:
                        if st.button("Cancel", key=f"cancel_rename_{draft_id}"):
                            st.session_state.draft_to_rename = None
                            st.rerun()

                    st.divider()

            # Render delete confirmation modal directly under this draft if it's selected
            if hasattr(st.session_state, 'draft_to_delete') and st.session_state.draft_to_delete == draft_id:
                with st.container():
                    st.subheader("Delete Draft")
                    st.warning(
                        "Are you sure you want to delete this draft? This action cannot be undone.")

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Yes, Delete", key=f"confirm_delete_{draft_id}"):
                            delete_draft(draft_id)
                            st.session_state.draft_to_delete = None
                            st.rerun()

                    with col2:
                        if st.button("Cancel", key=f"cancel_delete_{draft_id}"):
                            st.session_state.draft_to_delete = None
                            st.rerun()

                    st.divider()


def render_center_column():
    """Render the center column of the UI"""
    # Menu bar
    col1, col2 = st.columns([1, 1])

    with col1:
        # Save button
        if st.button("Save", key="save_button"):
            save_current_draft()

    with col2:
        # Approve button
        if st.button("Approve", key="approve_button"):
            approve_current_draft()

    # Text area for draft content
    draft_content = get_current_draft_content()
    updated_content = st.text_area(
        "Draft Content",
        value=draft_content,
        height=600,
        key="draft_content_area"
    )

    # Auto-save when content changes
    if updated_content != draft_content:
        save_draft_content(updated_content)


def render_right_column():
    """Render the right column of the UI (chat interface)"""
    st.subheader("AI Assistant")

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**Assistant:** {message['content']}")

    # Input for new messages
    user_input = st.text_input("Ask the AI Assistant", key="chat_input")

    if st.button("Send", key="send_message_button") and user_input:
        # Add user message to chat history
        SessionManager.add_chat_message('user', user_input)

        # Process the message and get AI response
        ai_response = process_ai_chat(user_input)
        SessionManager.add_chat_message('assistant', ai_response)

        # Clear the input
        st.session_state.chat_input = ""
        st.rerun()

    # Clear chat button
    if st.button("Clear Chat", key="clear_chat_button"):
        SessionManager.clear_chat_history()
        st.rerun()


def render_copy_modal():
    """Render the copy draft modal"""
    if hasattr(st.session_state, 'show_copy_modal') and st.session_state.show_copy_modal:
        with st.expander("Copy Draft", expanded=True):
            st.subheader("Copy Draft")

            draft_id = st.session_state.draft_to_copy
            current_title = get_draft_title(draft_id)
            new_title = st.text_input(
                "New Title", value=f"Copy of {current_title}", key="copy_draft_title")

            if st.button("Create Copy", key="create_copy_button") and new_title:
                copy_draft(draft_id, new_title)
                st.session_state.show_copy_modal = False
                st.session_state.draft_to_copy = None
                st.rerun()

            if st.button("Cancel", key="cancel_copy_button"):
                st.session_state.show_copy_modal = False
                st.session_state.draft_to_copy = None
                st.rerun()
