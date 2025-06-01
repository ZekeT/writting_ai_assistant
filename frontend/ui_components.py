from frontend.draft_operations import (
    get_drafts_for_current_publication,
    get_current_draft_content,
    save_draft_content,
    save_current_draft,
    approve_current_draft,
    create_new_draft,
    rename_draft,
    delete_draft,
    get_draft_title,
    process_ai_chat,
    get_articles_for_current_publication,
    toggle_article_inclusion,
    get_article_content
)
from backend.file_processor import process_uploaded_file
from backend.session_manager import SessionManager
from backend.database import get_db_instance
import streamlit as st
import sys
import os
from datetime import datetime

# Add the parent directory to the path to import backend modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Settings Panel


def render_settings_panel():
    """Render the settings panel"""
    with st.container():
        st.subheader("Writing Assistant Settings")

        # Example settings
        st.selectbox("AI Model", ["GPT-4", "GPT-3.5",
                     "Claude"], key="ai_model_setting")
        st.slider("Temperature", 0.0, 1.0, 0.7, 0.1, key="temperature_setting")
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

# Article Management


def render_articles_modal():
    """Render the articles management modal"""
    with st.container():
        st.subheader("Articles")

        # Get articles for current publication and date
        articles = get_articles_for_current_publication()

        if not articles:
            st.info("No articles found for this date and publication.")
        else:
            for article in articles:
                article_id = article['id']
                is_included = article['included'] == 1

                # Article row with buttons
                col1, col2 = st.columns([5, 1])

                with col1:
                    # Apply strikethrough and gray color if excluded
                    title_text = article['title']
                    if not is_included:
                        title_text = f"~~{title_text}~~"

                    # Use markdown for styling and button for click action
                    if st.button(
                        title_text,
                        key=f"article_{article_id}",
                        disabled=not is_included,
                        use_container_width=True
                    ):
                        # Set this article for preview
                        st.session_state.preview_article_id = article_id
                        # Close the modal when previewing
                        st.session_state.show_articles_modal = False
                        st.rerun()

                with col2:
                    # Toggle button - show X if included, checkmark if excluded
                    toggle_icon = "❌" if is_included else "✅"
                    toggle_help = "Exclude from context" if is_included else "Include in context"

                    if st.button(
                        toggle_icon,
                        key=f"toggle_{article_id}",
                        help=toggle_help
                    ):
                        # Toggle inclusion state
                        toggle_article_inclusion(article_id, not is_included)
                        st.rerun()

        # Close button
        if st.button("Close", key="close_articles_button"):
            st.session_state.show_articles_modal = False
            st.rerun()

        st.divider()


def render_upload_modal():
    """Render the file upload modal"""
    with st.container():
        st.subheader("Upload Article")

        uploaded_file = st.file_uploader(
            "Choose a file (PDF, DOCX, MD, TXT)",
            type=["pdf", "docx", "md", "txt"],
            key="article_file_uploader"
        )

        if uploaded_file is not None:
            # Show progress bar for processing
            progress_bar = st.progress(0)

            # Process file (extract content and convert to markdown)
            progress_bar.progress(25, text="Processing file...")

            title, content = process_uploaded_file(uploaded_file)
            progress_bar.progress(50, text="Extracting content...")

            if title is None:
                st.error(content)  # Show error message
            else:
                # Allow user to edit the title
                edited_title = st.text_input(
                    "Article Title", value=title, key="article_title_input")

                # Preview the content
                with st.expander("Content Preview", expanded=False):
                    st.markdown(content)

                progress_bar.progress(75, text="Ready to save...")

                # Save button
                if st.button("Save Article", key="save_article_button"):
                    # Get database instance
                    db = get_db_instance()

                    # Save article to database
                    publication_id = st.session_state.current_publication['id']
                    article_date = st.session_state.current_draft_date

                    article_id = db.create_article(
                        edited_title,
                        content,
                        publication_id,
                        article_date,
                        original_filename=uploaded_file.name
                    )

                    if article_id:
                        progress_bar.progress(100, text="Article saved!")
                        st.success(
                            f"Article '{edited_title}' saved successfully!")

                        # Close the modal after a short delay
                        st.session_state.show_upload_modal = False
                        st.rerun()
                    else:
                        st.error("Failed to save article. Please try again.")

        # Close button
        if st.button("Cancel", key="cancel_upload_button"):
            st.session_state.show_upload_modal = False
            st.rerun()

        st.divider()

# Draft Management


def render_add_draft_modal():
    """Render the add draft modal"""
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


def render_rename_draft_modal(draft_id):
    """Render the rename draft modal"""
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


def render_delete_draft_modal(draft_id):
    """Render the delete draft confirmation modal"""
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


def render_draft_list():
    """Render the draft list section"""
    st.subheader("Draft List")

    # Add new draft button as a "+" icon beside the header
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("➕", key="add_draft_icon", help="Add new draft"):
            st.session_state.show_add_draft_modal = True

    # Render add draft panel if needed
    if hasattr(st.session_state, 'show_add_draft_modal') and st.session_state.show_add_draft_modal:
        render_add_draft_modal()

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
                    # Clear any article preview when selecting a draft
                    if hasattr(st.session_state, 'preview_article_id'):
                        st.session_state.preview_article_id = None
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
                render_rename_draft_modal(draft_id)

            # Render delete confirmation modal directly under this draft if it's selected
            if hasattr(st.session_state, 'draft_to_delete') and st.session_state.draft_to_delete == draft_id:
                render_delete_draft_modal(draft_id)

# Main UI Components


def render_sidebar_content():
    """Render the sidebar content (former left column)"""
    # Settings icon
    if st.button("⚙️ Settings", key="settings_button"):
        SessionManager.toggle_settings()

    # Render settings panel directly under the settings button
    if st.session_state.show_settings:
        render_settings_panel()

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

    # Article management buttons (side by side)
    col1, col2 = st.columns(2)

    with col1:
        if st.button("View Articles", key="view_articles_button"):
            st.session_state.show_articles_modal = True
            # Reset article preview if we're showing the modal
            if hasattr(st.session_state, 'preview_article_id'):
                st.session_state.preview_article_id = None

    with col2:
        if st.button("Add Articles", key="add_articles_button"):
            st.session_state.show_upload_modal = True

    # Render articles modal directly under the buttons
    if hasattr(st.session_state, 'show_articles_modal') and st.session_state.show_articles_modal:
        render_articles_modal()

    # Render file upload modal
    if hasattr(st.session_state, 'show_upload_modal') and st.session_state.show_upload_modal:
        render_upload_modal()

    # Draft list section
    render_draft_list()


def render_article_preview(article_id):
    """Render article preview in the center column"""
    article_content = get_article_content(article_id)

    # Menu bar for article preview
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**Article Preview (Read-only)**")

    with col2:
        if st.button("Back to Draft", key="back_to_draft_button"):
            st.session_state.preview_article_id = None
            st.rerun()

    # Display article content (read-only)
    st.markdown(article_content)


def render_draft_editor():
    """Render the draft editor in the center column"""
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


def render_center_column():
    """Render the center column of the UI"""
    # Check if we're previewing an article
    if hasattr(st.session_state, 'preview_article_id') and st.session_state.preview_article_id:
        # Article preview mode
        render_article_preview(st.session_state.preview_article_id)
    else:
        # Normal draft editing mode
        render_draft_editor()


def render_chat_history():
    """Render the chat history in the right column"""
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**Assistant:** {message['content']}")


def render_chat_input():
    """Render the chat input in the right column"""
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


def render_right_column():
    """Render the right column of the UI (chat interface)"""
    st.subheader("AI Assistant")

    # Display chat history
    render_chat_history()

    # Input for new messages
    render_chat_input()

    # Clear chat button
    if st.button("Clear Chat", key="clear_chat_button"):
        SessionManager.clear_chat_history()
        st.rerun()
