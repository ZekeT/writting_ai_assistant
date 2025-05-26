# Writing Assistant Streamlit App

This is a Streamlit application designed to help investment strategists draft their publications. The app provides a three-column interface for managing drafts, editing content, and interacting with an AI assistant.

## Project Structure

```
writing_assistant_app/
├── backend/
│   ├── database.py         # SQLite database integration
│   ├── llm_interface.py    # LLM integration for AI assistance
│   └── session_manager.py  # Streamlit session state management
├── frontend/
│   ├── draft_operations.py # CRUD operations for drafts
│   └── ui_components.py    # UI components and layout
└── app.py                  # Main application entry point
```

## Features

- **Multi-publication Support**: Supports different publication types (Daily Wealth Wire, Weekly Investment Ideas, Wealth Focus)
- **Three-column Layout**:
  - Left column: Date picker, draft list, and settings
  - Center column: Draft content editor with menu bar
  - Right column: AI assistant chat interface
- **Draft Management**: Create, read, update, delete, and copy drafts
- **Session State**: Built-in per-user session state
- **Database Integration**: SQLite database for storing drafts (extensible to YugabyteDB)
- **AI Assistant**: Chat interface for interacting with an LLM to edit drafts

## Setup and Installation

1. Install the required dependencies:

```bash
pip install streamlit sqlite3
```

2. Run the application:

```bash
cd writing_assistant_app
streamlit run app.py
```

3. Setup DB

```bash
# Basic setup
python setup_database.py

# Add sample data for testing
python setup_database.py --sample-data

# Specify custom database path
python setup_database.py --db custom_path.db
```

## Usage

1. Select a publication type from the sidebar
2. Use the date picker to filter drafts by date
3. Create a new draft or select an existing one
4. Edit the draft content in the center column
5. Use the AI assistant in the right column for help with editing
6. Save, rename, copy, or approve drafts as needed

## Future Extensions

- **User Authentication**: Login and logout functionality
- **YugabyteDB Integration**: For scalable database storage
- **Advanced LLM Integration**: Connect to production LLM APIs for better assistance
- **Multi-user Collaboration**: Allow multiple users to work on the same draft

## Implementation Notes

- The current implementation uses placeholder functions for LLM integration
- The database is set up to support future authentication features
- The UI is designed to be responsive and user-friendly
- All draft operations are automatically saved to the database
