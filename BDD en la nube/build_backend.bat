rm -r backend_project
mkdir backend_project
node build_backend.js && cd backend_project && npm install && copy .env backend_project/.env