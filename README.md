# RL Codebase


## How to Use the Human Interface

### Components
The human interface consists of two components, one is a ReactJS front-end, the other one is a Python back-end.
- react\_frontend:
    - A React frontend allows us to use many useful off-the-shelf packages like Socket.io from npm
    - To build the frontend, under the frontend directory, run "npm run build", npm will build the project and place files in the corresponding directories under python\_backend, so that Flask can serve the website.
- python\_backend: 
    - A Python backend. It serves the frontend and communicates with it using efficient WebSocket.
    
### Compile and Build the Interface
#### The Frontend
- If node and npm are already installed, make sure they are updated to nearly latest version, this can be done by:
    ```sh
    sudo npm install -g n
    sudo n latest
    ```
    - Recommended version: "node >= v15.4.0" and "npm >= 7.0.15"
- If need to install node and npm (e.g. on our new server)
    ```sh
    sudo su
    sudo curl -sL https://rpm.nodesource.com/setup_15.x | bash -
    sudo yum install nodejs
    ```
  - Check the installed version via ``` node -v ``` and ``` npm -v ```
- After updating or installing node and npm:
    - For the first time, under folder react\_frontend, run ``` sudo npm install ```
    - Every time making change, under folder react\_frontend, run ``` sudo npm run build ```

#### The Backend
- On the Python side, the versions of packages matter a lot! Recommended commands:
    - Under directory \human\_interface, do:
    ```sh
    pip install -r requirements.txt
    ```
    - Or you can install each library one by one:
    ```sh
      pip install sanic==20.9.1
      pip install python-socketio==5.0.3
      pip install python-engineio==4.0.0
      pip install uvicorn==0.13.1
      pip install fastapi==0.62.0
    ```
    - If necessary, uninstall them and their dependencies, and reinstall
- Commands to setup and run
    - For the first time, under folder react\_frontend, run "\[sudo\] npm install"
    - Every time making change, under folder react\_frontend, run "\[sudo\] npm run build"
- Then under root directory of the project
    - To collect human demo or get action advice:
        - First run "python -m human_interface.flask_backend.app_control"
        - Then run the corresponding scrips (like "python -m scripts.collect\_demo --human" for demo collecting)
    
## Run the Interface
- We first need to start HTTP tunnel service to allow remote access to the website on "local" host (port 5000).
    - For example, if using ngrok, run ``` ngrok http 5000 ```, then copy the forwarding url (e.g. fbc114e8043c.ngrok.io).
- Then under the project root directory, start the python backend. Here we need to provide the forwarding url here (no leading protocol like http://). For example:
```sh
python -m scripts.collect_demo --human --host fbc114e8043c.ngrok.io
```
- Finally, access the interface via any modern browser.

## Rferences
- https://arxiv.org/pdf/1604.06057.pdf - Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction and Intrinsic Motivation
- https://arxiv.org/pdf/1804.08229v1.pdf - An Empirical Comparison of PDDL-based and ASP-based Task Planners
