from ..config import get_config
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from ..data.models import Base

config = get_config()

engine = create_engine(config['db_auth']['db_url'])
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)

def get_session():
    return Session()