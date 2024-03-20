FROM shankari/e-mission-server:master_2024-02-10--19-38

ENV DASH_DEBUG_MODE True
ENV SERVER_PORT 8050
ENV DASH_SERVER_PORT 8050
ENV DB_HOST db
ENV CONFIG_PATH "https://raw.githubusercontent.com/e-mission/nrel-openpath-deploy-configs/main/configs/"
ENV STUDY_CONFIG "stage-program"
ENV DASH_SILENCE_ROUTES_LOGGING False
ENV WEB_SERVER_HOST 0.0.0.0
ENV SERVER_BRANCH master
ENV REACT_VERSION "18.2.0"

# the other option is cognito
ENV AUTH_TYPE "basic" 

# copy over setup files
WORKDIR /usr/src/app/dashboard_setup
COPY requirements.txt nrel_dash_components-0.0.1.tar.gz docker/setup.sh ./

# install requirements.txt
WORKDIR /usr/src/app
RUN bash -c "./dashboard_setup/setup.sh"

# copy over dashboard code
WORKDIR /usr/src/app/pages
COPY ./pages ./
WORKDIR /usr/src/app/utils
COPY ./utils ./
WORKDIR /usr/src/app
COPY app.py config.py app_sidebar_collapsible.py assets globals.py globalsUpdater.py Procfile ./

WORKDIR /usr/src/app/assets
COPY assets/style.css ./
RUN mkdir qrcodes

# copy over test data
WORKDIR /usr/src/app/data
COPY data ./

# open listening port, this may be overridden in docker-compose file
EXPOSE ${SERVER_PORT}
EXPOSE ${DASH_SERVER_PORT}

# run the dashboard
WORKDIR /usr/src/app/dashboard_setup
COPY docker/start.sh ./
WORKDIR /usr/src/app
CMD ["/bin/bash", "/usr/src/app/dashboard_setup/start.sh"]
