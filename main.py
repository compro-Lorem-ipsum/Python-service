from app import create_app
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

app = create_app()