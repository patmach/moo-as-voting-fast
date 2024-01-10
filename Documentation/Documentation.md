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
  - /getRecommendations/<user_id>

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

- Example of request data

  - ```json
    {
    	'whiteListItemIDs': [32, 47, 253, 266, 555, 1061, 1619, 2340, 2959, 2997, 3418, 4011, 4161, 4901, 4963, 7458, 8984, 33679, 46723, 53322, 55363, 61323, 64957, 68157, 81564, 86898, 89492, 103249, 105844, 115210, 148626, 187593, 202429],
    	'blackListItemIDs': [189363, 93840, 1682, 590, 1580, 2953, 95441, 1, 45722, 5349, 96079, 49272, 108190],
        'currentListItemIDs': [],
        'count': 15,
        'metrics': [33, 26, 20, 13, 6],
        'metricVariantsCodes': ['also_negative', 'maximal_diversity', 'intra_list_distance_based_novelty', 'avg_ratings', '']
    }
    ```

    

- Response (in JSON)
  - Dictionary
    - Keys = item ID (Integer)
    - Values = metric scores In order: relevance, diversity, novelty, popularity, calibration (Array)

- Example of response data

  - ```json
    {
        "168": [95.71171602775985, 90.08456836777941, 62.680255494616, 10.737544026390541, 83.19340759715217],
    	"160080": [94.70690543417534, 86.10188818703168, 70.41777030835512, 23.193101704706365, 39.21020047427453], 
        "27773": [99.71952522566625, 90.93282368158113, 1.476083275329402, 88.42896025155231, 94.51327923054002], 
        "94018": [94.64414362384046, 84.81483799958659, 58.86083472131182, 30.756441368076377, 34.69852102567428], 
        "3979": [87.68170424540077, 87.15829205637759, 53.97749593454705, 43.73617016040326, 20.496661704893988], 
        "60397": [97.31801155313336, 85.87358812670686, 34.096195458569525, 48.736968725374076, 94.68283336366451], 
        "95558": [91.63793814711464, 97.67549723992501, 68.7215793929318, 9.350243532378876, 98.89009430388575], 
        "55577": [87.9570011069638, 91.26715900713194, 64.42218498251766, 22.823267980484623, 97.57380037878883], 
        "204698": [98.1984015886596, 99.71628909747466, 65.881612128059, 1.580463602815664, 85.23852968622566], 
        "2393": [92.491029176663, 96.6300721463742, 54.10505506159712, 25.419011549960217, 89.43264999360598], 
        "100498": [86.1471291571386, 84.57454407043521, 79.54912282734999, 16.67257131503161, 55.89537297504624], 
        "788": [94.43437957463668, 78.63961422770002, 23.402628781946795, 70.33597847370129, 98.00451946878945], 
        "193": [86.95403350949059, 98.84893665372616, 81.93941764211442, 7.598383831758807, 19.764971554860637], 
        "60950": [94.12007165888708, 94.11694461263582, 23.557400032800015, 54.69261906835441, 86.94939033589546], 
        "6157": [99.25873670128004, 66.93127685838842, 30.06585228955807, 67.0191718964582, 23.561599383964623]}
    
    ```

    




## Local run

### Run the RS

- File `app.py`
- `gunicorn app:app -w 1 --threads 8 -b 0.0.0.0:5000`
  - `-w` number of workers (every worker will train the RS before it's able to respond to queries)
  - `--threads` number of threads used by one worker

- Should be available at http://localhost:5000/

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



