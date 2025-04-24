"""
Example demonstrating the memory-enhanced agent in Agentor.

This example shows how to use the memory-enhanced agent pattern:
- Creating a memory-enhanced agent with different memory configurations
- Adding memories to the agent's long-term memory
- Recalling memories based on queries
- Using memories to provide context for agent decisions
"""

import asyncio
import logging
import time
import random
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from agentor.agents.memory_enhanced_agent import MemoryEnhancedAgent, MemoryEntry
from agentor.core.interfaces.tool import ITool, ToolResult

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PersonalAssistantAgent(MemoryEnhancedAgent):
    """A memory-enhanced personal assistant agent."""

    def __init__(self, name=None, memory_config=None):
        super().__init__(
            name=name or "PersonalAssistant",
            memory_config=memory_config or {
                "retention_days": 30,
                "capacity": 1000,
                "importance_threshold": 0.3
            }
        )
        self._setup_actions()

    def _setup_actions(self):
        """Set up the actions for this agent."""
        self.register_action('greet_user', self._greet_user)
        self.register_action('provide_recommendation', self._provide_recommendation)
        self.register_action('answer_question', self._answer_question)
        self.register_action('summarize_day', self._summarize_day)

    async def _greet_user(self) -> str:
        """Greet the user based on their preferences and history."""
        # Get the user's name from context
        user_name = self.state.get('current_context', {}).get('user_name', 'User')
        
        # Try to recall user preferences
        preferences = await self.recall(f"{user_name} preferences", limit=3)
        greeting_prefs = [p for p in preferences if "greeting" in p.content.lower()]
        
        # Try to recall last interaction
        last_interactions = await self.recall(f"last interaction with {user_name}", limit=1)
        
        # Construct greeting
        greeting = f"Hello, {user_name}!"
        
        # Add preference-based customization
        if greeting_prefs:
            pref = greeting_prefs[0].content
            if "formal" in pref.lower():
                greeting = f"Good day, {user_name}."
            elif "casual" in pref.lower():
                greeting = f"Hey {user_name}! What's up?"
        
        # Add context from last interaction
        if last_interactions:
            last_interaction = last_interactions[0].content
            if "follow up" in last_interaction.lower():
                greeting += " I remember we were discussing something important last time."
        
        # Add time-based customization
        current_hour = datetime.now().hour
        if 5 <= current_hour < 12:
            greeting = f"Good morning, {user_name}!"
        elif 12 <= current_hour < 17:
            greeting = f"Good afternoon, {user_name}!"
        elif 17 <= current_hour < 22:
            greeting = f"Good evening, {user_name}!"
        
        return greeting

    async def _provide_recommendation(self) -> str:
        """Provide a recommendation based on user preferences."""
        # Get the user's name from context
        user_name = self.state.get('current_context', {}).get('user_name', 'User')
        
        # Try to recall user preferences
        preferences = await self.recall(f"{user_name} preferences", limit=5)
        likes = [p for p in preferences if "likes" in p.content.lower()]
        dislikes = [p for p in preferences if "dislikes" in p.content.lower()]
        
        # Default recommendations
        recommendations = [
            "a new book on technology trends",
            "trying the new Italian restaurant downtown",
            "taking a short walk to clear your mind",
            "checking out the latest documentary on streaming services",
            "organizing your digital files for better productivity"
        ]
        
        # Personalize recommendations based on preferences
        if likes:
            like_content = likes[0].content.lower()
            if "book" in like_content or "reading" in like_content:
                recommendations.append("the latest bestseller in your favorite genre")
            if "movie" in like_content or "film" in like_content:
                recommendations.append("the new film that just won several awards")
            if "food" in like_content or "restaurant" in like_content:
                recommendations.append("that new fusion restaurant everyone's talking about")
            if "exercise" in like_content or "fitness" in like_content:
                recommendations.append("trying a new workout routine to mix things up")
        
        # Filter out recommendations based on dislikes
        if dislikes:
            dislike_content = dislikes[0].content.lower()
            recommendations = [r for r in recommendations if not any(d in r.lower() for d in dislike_content.split())]
        
        # Choose a random recommendation
        if recommendations:
            recommendation = random.choice(recommendations)
            return f"Based on what I know about you, I recommend {recommendation}."
        else:
            return "I don't have enough information about your preferences yet. Would you like to tell me more about what you enjoy?"

    async def _answer_question(self) -> str:
        """Answer a question using the agent's memory."""
        # Get the query from context
        query = self.state.get('current_query', '')
        
        if not query:
            return "I'm not sure what you're asking. Could you please rephrase your question?"
        
        # Try to recall relevant information
        relevant_info = await self.recall(query, limit=3)
        
        if relevant_info:
            # Construct an answer based on recalled information
            answer = "Based on what I remember: "
            for info in relevant_info:
                answer += f"\n- {info.content}"
            return answer
        else:
            return "I don't have any information about that in my memory. Would you like me to learn more about this topic?"

    async def _summarize_day(self) -> str:
        """Summarize the day's interactions."""
        # Get today's date
        today = datetime.now().date()
        
        # Try to recall today's interactions
        today_interactions = await self.recall(
            f"interactions on {today.isoformat()}",
            limit=10,
            metadata_filters={"date": today.isoformat()}
        )
        
        if today_interactions:
            # Construct a summary
            summary = f"Here's a summary of our interactions today ({today.strftime('%A, %B %d')}):\n"
            for i, interaction in enumerate(today_interactions, 1):
                summary += f"\n{i}. {interaction.content}"
            return summary
        else:
            return f"We haven't had any significant interactions today ({today.strftime('%A, %B %d')}) that I can recall."

    def decide(self) -> str:
        """Make a decision based on the current state.

        Returns:
            The name of the action to take
        """
        # Get the query from context
        query = self.state.get('current_query', '').lower()
        
        # Choose action based on query content
        if not query or "hello" in query or "hi" in query or "hey" in query:
            return "greet_user"
        elif "recommend" in query or "suggest" in query or "what should" in query:
            return "provide_recommendation"
        elif "summary" in query or "recap" in query or "what did we" in query:
            return "summarize_day"
        else:
            return "answer_question"


class CalendarTool(ITool):
    """A tool for managing calendar events."""

    @property
    def name(self) -> str:
        return "calendar"

    @property
    def description(self) -> str:
        return "Manage calendar events"

    async def run(self, action: str, **kwargs) -> ToolResult:
        """Perform a calendar action.
        
        Args:
            action: The action to perform (add_event, get_events, etc.)
            **kwargs: Additional parameters for the action
            
        Returns:
            Result of the calendar action
        """
        if action == "add_event":
            return await self._add_event(**kwargs)
        elif action == "get_events":
            return await self._get_events(**kwargs)
        else:
            return ToolResult(
                success=False,
                error=f"Unknown calendar action: {action}"
            )

    async def _add_event(self, title: str, date: str, time: str = None, duration: int = 60) -> ToolResult:
        """Add an event to the calendar.
        
        Args:
            title: The title of the event
            date: The date of the event (YYYY-MM-DD)
            time: The time of the event (HH:MM)
            duration: The duration of the event in minutes
            
        Returns:
            Result of adding the event
        """
        # In a real implementation, this would add to a real calendar
        # For this example, we'll just return a success message
        event_id = f"evt-{int(time.time())}"
        
        return ToolResult(
            success=True,
            data={
                "event_id": event_id,
                "title": title,
                "date": date,
                "time": time,
                "duration": duration,
                "message": f"Added event '{title}' to calendar on {date} at {time or 'all day'}"
            }
        )

    async def _get_events(self, date: str = None, days: int = 1) -> ToolResult:
        """Get events from the calendar.
        
        Args:
            date: The start date (YYYY-MM-DD)
            days: Number of days to include
            
        Returns:
            Result with calendar events
        """
        # In a real implementation, this would query a real calendar
        # For this example, we'll return mock events
        if not date:
            date = datetime.now().date().isoformat()
        
        # Parse the date
        try:
            start_date = datetime.fromisoformat(date).date()
        except ValueError:
            return ToolResult(
                success=False,
                error=f"Invalid date format: {date}. Use YYYY-MM-DD."
            )
        
        # Generate mock events
        events = []
        for i in range(days):
            current_date = start_date + timedelta(days=i)
            # Add 0-3 events per day
            for j in range(random.randint(0, 3)):
                hour = random.randint(9, 17)
                events.append({
                    "event_id": f"evt-{current_date.isoformat()}-{j}",
                    "title": random.choice([
                        "Team Meeting",
                        "Doctor Appointment",
                        "Lunch with Client",
                        "Project Review",
                        "Gym Session",
                        "Coffee Break"
                    ]),
                    "date": current_date.isoformat(),
                    "time": f"{hour:02d}:00",
                    "duration": random.choice([30, 60, 90, 120])
                })
        
        return ToolResult(
            success=True,
            data={
                "events": events,
                "start_date": start_date.isoformat(),
                "days": days,
                "count": len(events)
            }
        )

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add_event", "get_events"],
                    "description": "The calendar action to perform"
                },
                "title": {
                    "type": "string",
                    "description": "The title of the event (for add_event)"
                },
                "date": {
                    "type": "string",
                    "description": "The date in YYYY-MM-DD format"
                },
                "time": {
                    "type": "string",
                    "description": "The time in HH:MM format (for add_event)"
                },
                "duration": {
                    "type": "integer",
                    "description": "The duration in minutes (for add_event)"
                },
                "days": {
                    "type": "integer",
                    "description": "Number of days to include (for get_events)"
                }
            },
            "required": ["action"]
        }


class EnhancedAssistantAgent(PersonalAssistantAgent):
    """An enhanced personal assistant with tool integration and better memory."""

    def __init__(self, name=None):
        # Use a more sophisticated memory configuration
        memory_config = {
            "retention_days": 90,  # Remember for 3 months
            "capacity": 5000,      # Store up to 5000 memories
            "importance_threshold": 0.2,  # Lower threshold to remember more
            "vector_db": {
                "type": "in_memory",  # Use in-memory vector DB for this example
                "params": {}
            },
            "embedding": {
                "provider": "mock",  # Use mock embeddings for this example
                "dimension": 384
            }
        }
        
        super().__init__(name=name or "EnhancedAssistant", memory_config=memory_config)
        
        # Register the calendar tool
        self.calendar_tool = CalendarTool()
        self.register_tool(self.calendar_tool)
        
        # Register additional actions
        self.register_action('check_calendar', self._check_calendar)
        self.register_action('add_calendar_event', self._add_calendar_event)

    async def _check_calendar(self) -> str:
        """Check the calendar for upcoming events."""
        try:
            # Get today's date
            today = datetime.now().date().isoformat()
            
            # Get events for the next 3 days
            result = await self.calendar_tool.run(action="get_events", date=today, days=3)
            
            if not result.success:
                return f"I couldn't check your calendar: {result.error}"
            
            events = result.data.get("events", [])
            
            if not events:
                return "You don't have any events scheduled for the next 3 days."
            
            # Group events by date
            events_by_date = {}
            for event in events:
                date = event["date"]
                if date not in events_by_date:
                    events_by_date[date] = []
                events_by_date[date].append(event)
            
            # Format the response
            response = "Here's your schedule for the next few days:\n"
            
            for date, day_events in sorted(events_by_date.items()):
                # Format the date
                try:
                    date_obj = datetime.fromisoformat(date).date()
                    date_str = date_obj.strftime("%A, %B %d")
                except ValueError:
                    date_str = date
                
                response += f"\n{date_str}:"
                
                # Add events for this day
                for event in sorted(day_events, key=lambda e: e.get("time", "00:00")):
                    time_str = event.get("time", "All day")
                    duration = event.get("duration", 0)
                    duration_str = f" ({duration} min)" if duration else ""
                    response += f"\n  - {time_str}: {event['title']}{duration_str}"
            
            # Store this information in memory
            await self.add_memory(
                content=f"Calendar checked on {datetime.now().isoformat()}: {len(events)} events in the next 3 days",
                source="calendar",
                importance=0.6,
                metadata={"type": "calendar_check", "event_count": len(events)}
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error checking calendar: {str(e)}")
            return "I encountered an error while checking your calendar."

    async def _add_calendar_event(self) -> str:
        """Add an event to the calendar based on the current query."""
        try:
            # Parse the query for event details
            query = self.state.get('current_query', '')
            
            # Very basic parsing - in a real implementation, this would use NLP
            title = None
            date = None
            time = None
            
            # Extract title (anything after "add" and before "on" or "at")
            if "add" in query.lower():
                title_part = query.lower().split("add")[1].strip()
                if "on" in title_part:
                    title = title_part.split("on")[0].strip()
                elif "at" in title_part:
                    title = title_part.split("at")[0].strip()
                else:
                    title = title_part
            
            # Extract date (anything after "on" and before "at")
            if "on" in query.lower():
                date_part = query.lower().split("on")[1].strip()
                if "at" in date_part:
                    date = date_part.split("at")[0].strip()
                else:
                    date = date_part
            
            # Extract time (anything after "at")
            if "at" in query.lower():
                time = query.lower().split("at")[1].strip()
            
            # If we couldn't parse the details, ask for them
            if not title:
                return "What's the title of the event you want to add?"
            
            if not date:
                # Default to today
                date = datetime.now().date().isoformat()
            else:
                # Try to parse the date - this is very basic
                try:
                    # Check for common date formats
                    if "tomorrow" in date:
                        date = (datetime.now().date() + timedelta(days=1)).isoformat()
                    elif "today" in date:
                        date = datetime.now().date().isoformat()
                    # Add more date parsing as needed
                except Exception:
                    return f"I couldn't understand the date '{date}'. Please use a format like 'tomorrow' or 'today'."
            
            # Add the event
            result = await self.calendar_tool.run(
                action="add_event",
                title=title,
                date=date,
                time=time
            )
            
            if not result.success:
                return f"I couldn't add the event to your calendar: {result.error}"
            
            # Store this in memory
            await self.add_memory(
                content=f"Added calendar event: {title} on {date} at {time or 'all day'}",
                source="calendar",
                importance=0.7,
                metadata={"type": "calendar_add", "event_title": title, "event_date": date}
            )
            
            return result.data.get("message", "Event added to your calendar.")
            
        except Exception as e:
            logger.error(f"Error adding calendar event: {str(e)}")
            return "I encountered an error while adding the event to your calendar."

    def decide(self) -> str:
        """Make a decision based on the current state.

        Returns:
            The name of the action to take
        """
        # Get the query from context
        query = self.state.get('current_query', '').lower()
        
        # Choose action based on query content
        if "calendar" in query and "check" in query:
            return "check_calendar"
        elif "add" in query and ("event" in query or "calendar" in query or "schedule" in query):
            return "add_calendar_event"
        else:
            # Use the parent class decision logic
            return super().decide()


async def run_personal_assistant_example():
    """Run the personal assistant example."""
    logger.info("=== Personal Assistant Example ===")
    
    # Create the agent
    agent = PersonalAssistantAgent()
    
    # Add some initial memories
    await agent.add_memory(
        content="User prefers casual greetings",
        source="user",
        importance=0.7,
        metadata={"type": "preference", "category": "greeting"}
    )
    
    await agent.add_memory(
        content="User likes science fiction books",
        source="user",
        importance=0.6,
        metadata={"type": "preference", "category": "entertainment"}
    )
    
    await agent.add_memory(
        content="User dislikes horror movies",
        source="user",
        importance=0.6,
        metadata={"type": "preference", "category": "entertainment"}
    )
    
    await agent.add_memory(
        content="Last interaction with user: Promised to follow up on book recommendations",
        source="agent",
        importance=0.5,
        metadata={"type": "interaction", "category": "follow-up"}
    )
    
    # Run the agent with different queries
    queries = [
        "Hello there",
        "Can you recommend something for me?",
        "What do I like to read?",
        "What's a summary of our day?"
    ]
    
    for query in queries:
        logger.info(f"\nQuery: {query}")
        
        # Set up context
        context = {"user_name": "Alex"}
        
        # Run the agent
        result = await agent.run(query=query, context=context)
        
        logger.info(f"Response: {result.response}")
        
        # Wait a bit
        await asyncio.sleep(1)


async def run_enhanced_assistant_example():
    """Run the enhanced assistant example."""
    logger.info("\n=== Enhanced Assistant Example ===")
    
    # Create the agent
    agent = EnhancedAssistantAgent()
    
    # Add some initial memories
    await agent.add_memory(
        content="User prefers formal greetings",
        source="user",
        importance=0.7,
        metadata={"type": "preference", "category": "greeting"}
    )
    
    await agent.add_memory(
        content="User likes to exercise in the morning",
        source="user",
        importance=0.6,
        metadata={"type": "preference", "category": "routine"}
    )
    
    await agent.add_memory(
        content="User has a standing meeting every Monday at 10:00",
        source="calendar",
        importance=0.8,
        metadata={"type": "calendar", "category": "recurring"}
    )
    
    # Run the agent with different queries
    queries = [
        "Good morning",
        "Check my calendar please",
        "Add team lunch on tomorrow at 12:30",
        "What do I like to do in the morning?",
        "Check my calendar again"
    ]
    
    for query in queries:
        logger.info(f"\nQuery: {query}")
        
        # Set up context
        context = {"user_name": "Taylor"}
        
        # Run the agent
        result = await agent.run(query=query, context=context)
        
        logger.info(f"Response: {result.response}")
        
        # Wait a bit
        await asyncio.sleep(1)


async def main():
    """Run the memory-enhanced agent examples."""
    # Run the personal assistant example
    await run_personal_assistant_example()
    
    # Run the enhanced assistant example
    await run_enhanced_assistant_example()


if __name__ == "__main__":
    asyncio.run(main())
