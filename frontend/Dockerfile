# Step 1: Build the app
FROM node:16 as build

# Set working directory
WORKDIR /app

# Copy the package.json and install dependencies
COPY package.json package-lock.json ./
RUN npm install

# Copy the rest of the project files
COPY . ./

# Build the React app
RUN npm run build

# Step 2: Serve the built app using a lightweight web server
FROM nginx:alpine

# Copy the build output from the previous step to the nginx html directory
COPY --from=build /app/build /usr/share/nginx/html

# Expose port 80 (default for nginx)
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
