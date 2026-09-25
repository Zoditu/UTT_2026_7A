rm -r frontend_project
mkdir frontend_project
node build_frontend.js && cd frontend_project && npm install && copy .env frontend_project/.env