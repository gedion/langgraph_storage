import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

import pytest
from unittest.mock import AsyncMock, patch
from langgraph_storage import database

@pytest.mark.asyncio
async def test_start_and_stop_pool():
    """Test that start_pool and stop_pool execute without errors."""

    with patch.object(database, "start_pool", new_callable=AsyncMock) as mock_start, \
         patch.object(database, "stop_pool", new_callable=AsyncMock) as mock_stop:

        # Call the functions
        await database.start_pool()
        await database.stop_pool()

        # Ensure they were awaited once
        mock_start.assert_awaited_once()
        mock_stop.assert_awaited_once()
