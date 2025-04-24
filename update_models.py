"""
Update the models.py file to uncomment the training-related models and fix the circular dependencies.
"""

import os
import sys

# Add the current directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Read the models.py file
models_path = os.path.join("bo", "backend", "db", "models.py")
with open(models_path, "r") as f:
    content = f.read()

# Replace the commented out training-related models with the uncommented version
new_content = content.replace(
    """# Training-related models - commented out for now to fix circular dependencies
'''
class TrainingDataset(Base):
    # Dataset database model for training.
    __tablename__ = "datasets"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String, nullable=True)
    format = Column(String)  # csv, json, etc.
    size = Column(Integer)  # Size in bytes
    metadata = Column(String, nullable=True)  # JSON metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    creator_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Relationships
    creator = relationship("User")


class TrainingSession(Base):
    # Training Session database model.
    __tablename__ = "training_sessions"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=True)
    dataset_id = Column(String, ForeignKey("datasets.id"))
    config = Column(String)  # JSON configuration
    metrics = Column(String, nullable=True)  # JSON metrics
    status = Column(String)  # idle, running, completed, failed, stopped
    start_time = Column(DateTime(timezone=True))
    end_time = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    agent = relationship("Agent")
    dataset = relationship("TrainingDataset")


class TrainingModel(Base):
    # ML Model database model for training.
    __tablename__ = "models"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String, nullable=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=True)
    training_session_id = Column(String, ForeignKey("training_sessions.id"), nullable=True)
    config = Column(String)  # JSON configuration
    metrics = Column(String, nullable=True)  # JSON metrics
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    
    # Relationships
    agent = relationship("Agent")
    training_session = relationship("TrainingSession")
'''""",
    """# Training-related models
class TrainingDataset(Base):
    \"\"\"Dataset database model for training.\"\"\"
    __tablename__ = "datasets"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String, nullable=True)
    format = Column(String)  # csv, json, etc.
    size = Column(Integer)  # Size in bytes
    metadata = Column(String, nullable=True)  # JSON metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    creator_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Relationships
    creator = relationship("User", backref="datasets")


class TrainingSession(Base):
    \"\"\"Training Session database model.\"\"\"
    __tablename__ = "training_sessions"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=True)
    dataset_id = Column(String, ForeignKey("datasets.id"))
    config = Column(String)  # JSON configuration
    metrics = Column(String, nullable=True)  # JSON metrics
    status = Column(String)  # idle, running, completed, failed, stopped
    start_time = Column(DateTime(timezone=True))
    end_time = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    agent = relationship("Agent", backref="training_sessions")
    dataset = relationship("TrainingDataset", backref="training_sessions")


class TrainingModel(Base):
    \"\"\"ML Model database model for training.\"\"\"
    __tablename__ = "models"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String, nullable=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=True)
    training_session_id = Column(String, ForeignKey("training_sessions.id"), nullable=True)
    config = Column(String)  # JSON configuration
    metrics = Column(String, nullable=True)  # JSON metrics
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    
    # Relationships
    agent = relationship("Agent", backref="models")
    training_session = relationship("TrainingSession", backref="models")"""
)

# Write the updated content back to the file
with open(models_path, "w") as f:
    f.write(new_content)

print("Models updated successfully!")

# Update the init_data.py file to uncomment the create_default_datasets call
init_data_path = os.path.join("bo", "backend", "db", "init_data.py")
with open(init_data_path, "r") as f:
    content = f.read()

# Replace the commented out create_default_datasets call with the uncommented version
new_content = content.replace(
    "# create_default_datasets(db)",
    "create_default_datasets(db)"
)

# Write the updated content back to the file
with open(init_data_path, "w") as f:
    f.write(new_content)

print("Init data updated successfully!")
