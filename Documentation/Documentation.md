# Documentation - RS

## Set connection string

Go to file `db_connect.py` set properties, based on where the code and the MS SQL Server is running

#### Server name

- "np:\\\\.\\pipe\LOCALDB#{servernamecode}\\tsql\\query"
  - When connecting to local mssql server and code is run locally
  - `servernamecode` can be retrieved from `select @@servername` command in the database
- "localhost,{port}"
  - when code i run locally and MS SQL server is run in local container
  - `port` depends on port number set in dockerfile
- "{nameofcontainer}"
  - when code and sql server is run in containers on the same container network
  - `nameofcontainer` is based on name of the container with sql server

#### Driver name

- "SQL Server"
  - Running code locally on Windows
- "ODBC Driver 18 for SQL Server"
  - Running code in container (from python:3.9.10-slim-buster)
- Other
  - If code run differently 

#### Database name

- Name of the database used
  - Needs to contain table Ratings with columns userid, itemid, ratingscore, date

#### Username

- Username for connection to the database

#### Password

- Password for connection to the database



## Call API

- Route
  - "/getRecommendations/<user_id>"
- Parameters (in JSON)
  - `WhiteListItemIDs` - IDs of items possible to recommend. Empty means all are possible (Array)
  - `BlackListItemIDs` - IDs of items that shouldn't be returned (Array)
  - `CurrentListItemIDs` - IDs of items that are already part of displayed recommendations (Array)
  - `Count` - Number of items that should be returned (Integer)
  - `Metrics` - Metrics importance (=weights) specified by user. In order: relevance, diversity, novelty, popularity, calibration(Array)
  - `MetricVariantsCodes` - Metric variants used by user. In order: relevance, diversity, novelty, popularity (Array)
    - Currently possible values
      - Relevance
        - `only_positive`
        - `also_negative`
      - Diversity
        - `intra_list_diversity`
        - `maximal_diversity`
        - `binomial_diversity`
      - Novelty
        - `popularity_complement`
        - `maximal_distance_based_novelty`
        - `intra_list_distance_based_novelty`
      - Popularity
        - `num_of_ratings`
        - `avg_ratings`
- Response (in JSON)
  - Dictionary
    - Keys = item ID (Integer)
    - Values = metric scores In order: relevance, diversity, novelty, popularity, calibration (Array)



## Local run

### Run the RS

- File `app.py`
- `gunicorn app:app -w 1 --threads 8 -b 0.0.0.0:5000`
  - `-w` number of workers (every worker will train the RS before it's able to respond to queries)
  - `--threads` number of threads used by one worker

- Should be available at `http://localhost:5000/

### Run simulation of recommending to check dependencies

- Recommend to existing users (1000 randomly sampled and take into account their first random number of ratings from 3 to 50) to check dependency of normalized scores of metrics on rank in the list of recommendations and number of rated items by user 

  - File `recommending_simulation.py `

- Visualize the dependencies 

  - File `graphs_simulation.py`
    - Set `file` variable if needed

- ```shell
  python recommending_simulation.py 
  python graphs_simulation.py
  ```

- Check directory `NormalizationGraphs`

### Visualize results from user study

- ```shell
  python ResultsScript/results.py
  ```

- Check directory `Results`





## Run

- Add the `docker-compose.yml` and main directory of application to the same directory.

- Content of `docker-compose.yml`:

  - ```yaml
    version: "3.9"
    services:
      #sql-server-db:
        #Parameters to run MSSQL Server
        #If connected to remote MSSQL Server dont't use. Edit parameters in file db_connect.py
      rs:
        build: ./moo-as-voting-fast
        ports:
          - "5011:5000"
        #depends_on: #If connected to remote MSSQL Server dont't use. Edit parameters in file db_connect.py
        #  - sql-server-db
        container_name: rs
    
    ```

- Run `docker compose up` in shell

- Send a request to the URL of app (locally `localhost:5011`)

  - The RS training will be computed before response to the first request
  - After training is completed RS is ready to process requests on recommendations and send responses



