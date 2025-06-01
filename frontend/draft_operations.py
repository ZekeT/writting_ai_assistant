from backend.llm_interface import LLMInterface
from backend.session_manager import SessionManager
from backend.database import get_db_instance
import streamlit as st
import sys
import os
import json

# Add the parent directory to the path to import backend modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_drafts_for_current_publication():
    """Get drafts for the current publication and date"""
    db = get_db_instance()
    publication_id = st.session_state.current_publication['id']
    draft_date = st.session_state.current_draft_date
    # None for now, until login is implemented
    user_id = st.session_state.get('user_id')

    return db.get_drafts_by_date_and_publication(publication_id, draft_date, user_id)


def get_current_draft_content():
    """Get the content of the current draft"""
    if not hasattr(st.session_state, 'current_draft_id') or not st.session_state.current_draft_id:
        return ""

    db = get_db_instance()
    draft = db.get_draft_by_id(st.session_state.current_draft_id)

    if not draft:
        return ""

    return draft['content'] or ""


def save_draft_content(content):
    """Save the content of the current draft"""
    if not hasattr(st.session_state, 'current_draft_id') or not st.session_state.current_draft_id:
        return False

    db = get_db_instance()
    return db.update_draft(st.session_state.current_draft_id, content=content)


def save_current_draft():
    """Save the current draft explicitly"""
    if save_draft_content(st.session_state.draft_content_area):
        st.success("Draft saved successfully!")
    else:
        st.error("Failed to save draft.")


def approve_current_draft():
    """Mark the current draft as approved"""
    if not hasattr(st.session_state, 'current_draft_id') or not st.session_state.current_draft_id:
        st.error("No draft selected.")
        return False

    db = get_db_instance()
    if db.update_draft(st.session_state.current_draft_id, approved=True):
        st.success("Draft approved!")
        return True
    else:
        st.error("Failed to approve draft.")
        return False


def create_new_draft(title):
    """Create a new draft"""
    db = get_db_instance()
    publication_id = st.session_state.current_publication['id']
    draft_date = st.session_state.current_draft_date
    # None for now, until login is implemented
    user_id = st.session_state.get('user_id')

    # Get included articles content for context
    articles_context = db.get_included_articles_content(
        publication_id, draft_date)

    # Initialize LLM with context
    llm = LLMInterface()

    # Generate initial content if articles are available
    initial_content = ""
    memory_context = None

    if articles_context:
        # Use LLM to generate initial content based on articles
        initial_content, memory_context = llm.generate_draft(
            st.session_state.current_publication['name'],
            draft_date,
            articles_context
        )

    # Create the draft
    draft_id = db.create_draft(
        title, publication_id, draft_date, user_id, initial_content, memory_context
    )

    if draft_id:
        # Set as current draft
        SessionManager.set_current_draft(draft_id)
        st.success(f"Draft '{title}' created successfully!")
        return True
    else:
        st.error("Failed to create draft.")
        return False


def rename_draft(draft_id, new_title):
    """Rename a draft"""
    db = get_db_instance()
    if db.update_draft(draft_id, title=new_title):
        st.success(f"Draft renamed to '{new_title}'.")
        return True
    else:
        st.error("Failed to rename draft.")
        return False


def delete_draft(draft_id):
    """Delete a draft"""
    db = get_db_instance()
    if db.delete_draft(draft_id):
        # If the deleted draft was the current one, clear the current draft
        if hasattr(st.session_state, 'current_draft_id') and st.session_state.current_draft_id == draft_id:
            st.session_state.current_draft_id = None

        st.success("Draft deleted successfully.")
        return True
    else:
        st.error("Failed to delete draft.")
        return False


def copy_draft(draft_id, new_title):
    """Create a copy of a draft"""
    db = get_db_instance()
    original_draft = db.get_draft_by_id(draft_id)

    if not original_draft:
        st.error("Original draft not found.")
        return False

    # Create a new draft with the same content
    new_draft_id = db.create_draft(
        new_title,
        original_draft['publication_id'],
        original_draft['draft_date'],
        original_draft.get('user_id'),
        original_draft['content'],
        json.loads(original_draft['memory_context']) if original_draft.get(
            'memory_context') else None
    )

    if new_draft_id:
        # Set as current draft
        SessionManager.set_current_draft(new_draft_id)
        st.success(f"Draft copied as '{new_title}'.")
        return True
    else:
        st.error("Failed to copy draft.")
        return False


def get_draft_title(draft_id):
    """Get the title of a draft"""
    db = get_db_instance()
    draft = db.get_draft_by_id(draft_id)

    if not draft:
        return ""

    return draft['title']


def process_ai_chat(user_input):
    """Process a chat message with the AI assistant"""
    if not hasattr(st.session_state, 'current_draft_id') or not st.session_state.current_draft_id:
        return "Please select a draft first to chat with the AI assistant."

    db = get_db_instance()
    draft = db.get_draft_by_id(st.session_state.current_draft_id)

    if not draft:
        return "Error: Draft not found."

    # Get memory context from draft
    memory_context = json.loads(draft['memory_context']) if draft.get(
        'memory_context') else None

    # Use LLM interface with the memory context
    llm = LLMInterface()
    response, updated_memory = llm.chat_response(user_input, memory_context)

    # Update memory context in database
    db.update_draft(st.session_state.current_draft_id,
                    memory_context=updated_memory)

    return response

# Article-related functions


def get_articles_for_current_publication():
    """Get articles for the current publication and date"""
    db = get_db_instance()
    publication_id = st.session_state.current_publication['id']
    article_date = st.session_state.current_draft_date

    return db.get_articles_by_date_and_publication(publication_id, article_date)


def toggle_article_inclusion(article_id, included):
    """Toggle whether an article is included in the context"""
    db = get_db_instance()
    if db.update_article_inclusion(article_id, included):
        return True
    else:
        st.error("Failed to update article inclusion state.")
        return False


def get_article_content(article_id):
    """Get the content of an article"""
    db = get_db_instance()
    article = db.get_article_by_id(article_id)

    if not article:
        return "Article not found."

    return article['content']
