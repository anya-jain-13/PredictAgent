
# Importing FastAPI for creating the Notification API
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4


# Creating the Notification API application
app = FastAPI(title="PredictAgent Notification API")


# Creating a schema for incoming maintenance alerts
class NotificationRequest(BaseModel):
    equipment_id: str
    message: str = Field(min_length=1)
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    recipient: str = "maintenance_team"


# Creating temporary storage for simulated notifications
notifications = []


# Checking whether the Notification API is running
@app.get("/health")
def health():
    return {
        "status": "running",
        "service": "notification",
        "data_source": "simulated"
    }


# Receiving and recording a maintenance notification
@app.post("/notifications")
def create_notification(notification: NotificationRequest):

    # Creating a unique identifier for the notification
    notification_id = str(uuid4())

    # Recording the notification with its creation time
    record = {
        "notification_id": notification_id,
        "equipment_id": notification.equipment_id,
        "message": notification.message,
        "severity": notification.severity,
        "recipient": notification.recipient,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "SIMULATED",
        "delivery_status": "NOT_SENT"
    }

    # Storing the notification temporarily in memory
    notifications.append(record)

    # Returning confirmation that the notification was recorded
    return record


# Retrieving all recorded notifications
@app.get("/notifications")
def get_notifications():
    return {
        "count": len(notifications),
        "notifications": notifications,
        "data_source": "simulated"
    }


# Retrieving a specific notification
@app.get("/notifications/{notification_id}")
def get_notification(notification_id: str):

    # Searching for the requested notification
    for notification in notifications:
        if notification["notification_id"] == notification_id:
            return notification

    # Returning an error when the notification is not found
    raise HTTPException(
        status_code=404,
        detail="Notification not found"
    )
