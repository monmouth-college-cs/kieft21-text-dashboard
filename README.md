# Summer 2021 Kieft Research Project: Exploratory Text Analytics Dashboard

Students: Cal Bigham and Shay Hafner. 
Faculty advisor: Robert Utterback

## Setup

```bash
python3 -m venv --prompt kieft env # or similar
source env/bin/activate
pip install -r requirements.txt
./run-redis.sh
./start-celery.sh
python3 run.py
```

To run a production server, first install Waitress (`pip install
waitress`) (or the server of your choice). Then, instead of `python3
run.py`, run `waitress-serve --call 'app:create_app'`.
