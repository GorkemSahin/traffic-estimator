import axios from 'axios';

const axiosInstance = axios.create({
  baseURL: "http://192.168.0.31:8000",
  timeout: 10000000
});

export default axiosInstance;
