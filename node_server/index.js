const express = require("express");
const https = require("https");
const axios = require("axios");
const multer = require("multer");
const FormData = require("form-data");
const cors = require("cors");
require("dotenv").config();

const SERVER_IP="https://104.171.203.243:8775";

const app = express();
const upload = multer();

app.use(cors());
app.use(express.json());

app.post("/api/depth", upload.single("image"), async (req, res) => {
  try {
    const formData = new FormData();
    formData.append("image", req.file.buffer, "image.jpg");

    const agent = new https.Agent({ rejectUnauthorized: false });

    const response = await axios.post(
      `${SERVER_IP}/api/kmeans-depth`,
      formData,
      {
        headers: formData.getHeaders(),
        httpsAgent: agent,
      }
    );
    
    res.status(response.status).json(response.data);
  } catch (error) {
    console.error("Error forwarding the image:", error);
    res.status(500).json({ error: "Failed to process image" });
  }
});

const PORT = process.env.PORT || 4000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
