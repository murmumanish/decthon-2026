FROM node:24.13.0-alpine

WORKDIR /agent

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

EXPOSE 3000
CMD ["node", "index.js"]
