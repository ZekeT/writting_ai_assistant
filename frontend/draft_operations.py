from backend.llm_interface import LLMInterface, LLMProvider
from backend.session_manager import SessionManager
from backend.database import get_db_instance
import streamlit as st
import sys
import os
import json
from datetime import datetime

# Add the parent directory to the path to import backend modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_drafts_for_current_publication():
    """Get drafts for the current publication and date"""
    db = get_db_instance()
    publication_id = st.session_state.current_publication['id']
    draft_date = st.session_state.current_draft_date

    return db.get_drafts_by_date_and_publication(publication_id, draft_date)


def get_current_draft_content():
    """Get content of the current draft"""
    if not st.session_state.current_draft_id:
        return ""

    db = get_db_instance()
    draft = db.get_draft_by_id(st.session_state.current_draft_id)

    if not draft:
        return ""

    return draft.get('content', '')


def save_draft_content(content):
    """Save content of the current draft"""
    if not st.session_state.current_draft_id:
        return

    db = get_db_instance()
    db.update_draft(st.session_state.current_draft_id, content=content)


def save_current_draft():
    """Save the current draft"""
    if not st.session_state.current_draft_id:
        st.warning("No draft selected")
        return

    content = st.session_state.draft_content_area
    db = get_db_instance()
    success = db.update_draft(
        st.session_state.current_draft_id, content=content)

    if success:
        st.success("Draft saved successfully")
    else:
        st.error("Failed to save draft")


def approve_current_draft():
    """Approve the current draft"""
    if not st.session_state.current_draft_id:
        st.warning("No draft selected")
        return

    db = get_db_instance()
    success = db.update_draft(st.session_state.current_draft_id, approved=True)

    if success:
        st.success("Draft approved")
    else:
        st.error("Failed to approve draft")


def create_new_draft(title, provider=None):
    """Create a new draft with LangGraph memory"""
    db = get_db_instance()
    publication_id = st.session_state.current_publication['id']
    publication_name = st.session_state.current_publication['name']
    draft_date = st.session_state.current_draft_date

    # Initialize LLM interface with LangGraph memory and specified provider
    llm = LLMInterface(provider=provider)

    # Generate draft content with LangGraph memory
    try:
        # Generate initial draft content and memory
        content, memory_json = llm.generate_draft(
            publication_name,
            draft_date,
            f"Create a draft for {publication_name} on {draft_date}"
        )

        # Create draft in database with memory context
        draft_id = db.create_draft(
            title,
            publication_id,
            draft_date,
            content=content,
            memory_context=memory_json
        )

        if draft_id:
            st.success(
                f"Draft '{title}' created successfully using {llm.provider} provider")
            SessionManager.set_current_draft(draft_id)
            return True
        else:
            st.error("Failed to create draft")
            return False

    except Exception as e:
        st.error(f"Error creating draft: {str(e)}")
        return False


def rename_draft(draft_id, new_title):
    """Rename a draft"""
    db = get_db_instance()
    success = db.update_draft(draft_id, title=new_title)

    if success:
        st.success(f"Draft renamed to '{new_title}'")
        return True
    else:
        st.error("Failed to rename draft")
        return False


def delete_draft(draft_id):
    """Delete a draft"""
    db = get_db_instance()
    success = db.delete_draft(draft_id)

    if success:
        st.success("Draft deleted")
        # If the deleted draft was the current one, reset current draft
        if st.session_state.current_draft_id == draft_id:
            SessionManager.set_current_draft(None)
        return True
    else:
        st.error("Failed to delete draft")
        return False


def copy_draft(draft_id, new_title):
    """Copy a draft with its memory context"""
    db = get_db_instance()
    original_draft = db.get_draft_by_id(draft_id)

    if not original_draft:
        st.error("Original draft not found")
        return False

    publication_id = original_draft['publication_id']
    draft_date = original_draft['draft_date']
    content = original_draft['content']
    memory_context = original_draft['memory_context']

    # Create a copy with the same content and memory context
    new_draft_id = db.create_draft(
        new_title,
        publication_id,
        draft_date,
        content=content,
        memory_context=memory_context
    )

    if new_draft_id:
        st.success(f"Draft copied as '{new_title}'")
        SessionManager.set_current_draft(new_draft_id)
        return True
    else:
        st.error("Failed to copy draft")
        return False


def get_draft_title(draft_id):
    """Get the title of a draft"""
    db = get_db_instance()
    draft = db.get_draft_by_id(draft_id)

    if not draft:
        return ""

    return draft.get('title', '')


def process_ai_chat(user_input, provider=None):
    """Process chat with AI using LangGraph memory"""
    if not st.session_state.current_draft_id:
        return "Please select a draft first to enable chat functionality."

    db = get_db_instance()
    draft = db.get_draft_by_id(st.session_state.current_draft_id)

    if not draft:
        return "Error: Draft not found."

    try:
        # Get memory context from draft
        memory_context = draft.get('memory_context')

        # Initialize LLM interface with specified provider
        llm = LLMInterface(provider=provider)

        # Process chat with LangGraph memory
        response, updated_memory = llm.process_chat(user_input, memory_context)

        # Update draft with new memory context
        db.update_draft(st.session_state.current_draft_id,
                        memory_context=updated_memory)

        # Get the provider that was actually used (might be from memory)
        used_provider = llm.provider

        return f"{response}\n\n_Using {used_provider} provider_"

    except Exception as e:
        error_message = f"Error processing chat: {str(e)}"
        print(error_message)  # Log the error
        return f"I encountered an error while processing your request. Please try again or contact support if the issue persists."


def get_memory_summary(draft_id):
    """Get a summary of the LangGraph memory for a draft"""
    db = get_db_instance()
    draft = db.get_draft_by_id(draft_id)

    if not draft or not draft.get('memory_context'):
        return {
            "publication_type": "Unknown",
            "draft_date": "Unknown",
            "message_counts": {"total": 0, "human": 0, "ai": 0, "system": 0},
            "has_draft": False,
            "llm_provider": "Unknown"
        }

    try:
        # Initialize LLM interface
        llm = LLMInterface()

        # Get memory summary
        return llm.get_memory_summary(draft.get('memory_context'))

    except Exception as e:
        print(f"Error getting memory summary: {str(e)}")
        return {
            "error": str(e),
            "publication_type": "Error",
            "draft_date": "Error",
            "message_counts": {"total": 0, "human": 0, "ai": 0, "system": 0},
            "has_draft": False,
            "llm_provider": "Unknown"
        }


def get_available_providers():
    """Get available LLM providers"""
    llm = LLMInterface()
    return llm.get_available_providers()
