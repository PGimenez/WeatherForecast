<template>
  <div class="app">
    <div class="header">
      <h1>Weather Forecast</h1>
      <button 
        @click="downloadData" 
        class="download-btn"
        :disabled="downloading"
      >
        {{ downloading ? 'Downloading...' : 'Download Data' }}
      </button>
    </div>
    <div class="forecast-form">
      <div class="form-group">
        <label for="city">City:</label>
        <select 
          id="city" 
          v-model="city" 
          class="city-select"
        >
          <option value="">Select a city</option>
          <option 
            v-for="city in cities" 
            :key="city.name" 
            :value="city.name"
          >
            {{ city.name }}
          </option>
        </select>
      </div>
      <div class="form-group">
        <label for="startDate">Start Date:</label>
        <input 
          id="startDate" 
          v-model="startDate" 
          type="date"
        >
      </div>
      <div class="form-group">
        <label for="endDate">End Date:</label>
        <input 
          id="endDate" 
          v-model="endDate" 
          type="date"
        >
      </div>
    </div>

    <div v-if="error" class="error">
      {{ error }}
    </div>

    <div v-if="forecast && chartData" class="forecast-results">
      <LineChart 
        :data="chartData"
        :options="chartOptions"
      />
    </div>
  </div>
</template>

<script>
import axios from 'axios';
import { Line as LineChart } from 'vue-chartjs';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const API_URL = process.env.VITE_API_URL || 'http://localhost:8000';

export default {
  name: 'App',
  components: {
    LineChart
  },
  data() {
    return {
      city: 'Barcelona',
      startDate: '2024-08-01',
      endDate: '2024-12-31',
      forecast: null,
      loading: false,
      error: null,
      cities: [
        {"name": "Barcelona", "latitude": 41.3851, "longitude": 2.1734},
        {"name": "L'Hospitalet de Llobregat", "latitude": 41.3662, "longitude": 2.1169},
        {"name": "Badalona", "latitude": 41.4500, "longitude": 2.2474},
        {"name": "Terrassa", "latitude": 41.5610, "longitude": 2.0089},
        {"name": "Sabadell", "latitude": 41.5433, "longitude": 2.1094},
        {"name": "Lleida", "latitude": 41.6176, "longitude": 0.6200},
        {"name": "Tarragona", "latitude": 41.1189, "longitude": 1.2445},
        {"name": "Mataró", "latitude": 41.5381, "longitude": 2.4445},
        {"name": "Santa Coloma de Gramenet", "latitude": 41.4515, "longitude": 2.2080},
        {"name": "Reus", "latitude": 41.1498, "longitude": 1.1055}
      ],
      chartData: {
        labels: [],
        datasets: [
          {
            label: 'Predicted Temperature (°C)',
            data: [],
            borderColor: '#ff6384',
            backgroundColor: 'rgba(255, 99, 132, 0.1)',
            borderWidth: 2,
            tension: 0.3,
            yAxisID: 'y-temperature',
            pointRadius: 0
          },
          {
            label: 'Actual Temperature (°C)',
            data: [],
            borderColor: '#4bc0c0',
            backgroundColor: 'rgba(75, 192, 192, 0.1)',
            borderWidth: 2,
            tension: 0.3,
            yAxisID: 'y-temperature',
            borderDash: [5, 5],
            pointRadius: 0
          },
          {
            label: 'Predicted Precipitation (mm)',
            data: [],
            borderColor: '#36a2eb',
            backgroundColor: 'rgba(54, 162, 235, 0.1)',
            borderWidth: 2,
            tension: 0.3,
            yAxisID: 'y-precipitation',
            pointRadius: 0
          },
          {
            label: 'Actual Precipitation (mm)',
            data: [],
            borderColor: '#9966ff',
            backgroundColor: 'rgba(153, 102, 255, 0.1)',
            borderWidth: 2,
            tension: 0.3,
            yAxisID: 'y-precipitation',
            borderDash: [5, 5],
            pointRadius: 0
          }
        ]
      },
      downloading: false,
    };
  },
  computed: {
    chartOptions() {
      return {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: 'index',
          intersect: false,
        },
        plugins: {
          legend: {
            position: 'top',
            labels: {
              padding: 20,
              usePointStyle: true
            }
          },
          title: {
            display: true,
            text: `Weather Forecast for ${this.city}`,
            padding: 20,
            font: {
              size: 16
            }
          },
          tooltip: {
            mode: 'index',
            intersect: false,
            callbacks: {
              label: function(context) {
                let label = context.dataset.label || '';
                if (label) {
                  label += ': ';
                }
                if (context.parsed.y !== null) {
                  label += context.parsed.y.toFixed(1);
                }
                return label;
              }
            }
          }
        },
        scales: {
          x: {
            grid: {
              display: false
            },
            ticks: {
              maxRotation: 45,
              minRotation: 45,
              maxTicksLimit: 24,
              callback: function(value, index, values) {
                const date = new Date(this.getLabelForValue(value));
                const hour = date.getHours();
                const day = date.getDate();
                const month = date.toLocaleString('en-US', { month: 'short' });
                
                if (hour === 0 || index === 0 || index === values.length - 1) {
                  return `${month} ${day}, ${hour}:00`;
                }
                return `${hour}:00`;
              }
            }
          },
          'y-temperature': {
            type: 'linear',
            display: true,
            position: 'left',
            title: {
              display: true,
              text: 'Temperature (°C)'
            },
            grid: {
              borderDash: [2, 2]
            }
          },
          'y-precipitation': {
            type: 'linear',
            display: true,
            position: 'right',
            title: {
              display: true,
              text: 'Precipitation (mm)'
            },
            min: 0,
            grid: {
              display: false
            }
          }
        }
      };
    }
  },
  methods: {
    async getForecast() {
      if (!this.city || !this.startDate || !this.endDate) {
        this.error = 'Please fill in all fields';
        return;
      }

      this.loading = true;
      this.error = null;

      try {
        const response = await axios.post(`${API_URL}/predict`, {
          city: this.city,
          start_date: this.startDate,
          end_date: this.endDate
        }, {
          headers: {
            'Content-Type': 'application/json'
          }
        });
        
        this.forecast = response.data;
        this.chartData = {
          labels: this.forecast.dates,
          datasets: [
            {
              label: 'Predicted Temperature (°C)',
              data: this.forecast.temperature,
              borderColor: '#ff6384',
              backgroundColor: 'rgba(255, 99, 132, 0.1)',
              borderWidth: 2,
              tension: 0.3,
              yAxisID: 'y-temperature',
              pointRadius: 0
            },
            {
              label: 'Actual Temperature (°C)',
              data: this.forecast.actual_temperature,
              borderColor: '#4bc0c0',
              backgroundColor: 'rgba(75, 192, 192, 0.1)',
              borderWidth: 2,
              tension: 0.3,
              yAxisID: 'y-temperature',
              borderDash: [5, 5],
              pointRadius: 0
            },
            {
              label: 'Predicted Precipitation (mm)',
              data: this.forecast.precipitation,
              borderColor: '#36a2eb',
              backgroundColor: 'rgba(54, 162, 235, 0.1)',
              borderWidth: 2,
              tension: 0.3,
              yAxisID: 'y-precipitation',
              pointRadius: 0
            },
            {
              label: 'Actual Precipitation (mm)',
              data: this.forecast.actual_precipitation,
              borderColor: '#9966ff',
              backgroundColor: 'rgba(153, 102, 255, 0.1)',
              borderWidth: 2,
              tension: 0.3,
              yAxisID: 'y-precipitation',
              borderDash: [5, 5],
              pointRadius: 0
            }
          ]
        };
      } catch (err) {
        this.error = err.response?.data?.detail || 'An error occurred while fetching the forecast';
      } finally {
        this.loading = false;
      }
    },
    debounce(fn, delay) {
      let timeoutId;
      return function (...args) {
        if (timeoutId) {
          clearTimeout(timeoutId);
        }
        timeoutId = setTimeout(() => {
          fn.apply(this, args);
        }, delay);
      };
    },
    async downloadData() {
      this.downloading = true;
      try {
        const response = await axios({
          url: `${API_URL}/download-data`,
          method: 'GET',
          responseType: 'blob'
        });
        
        // Create a blob URL and trigger download
        const url = window.URL.createObjectURL(new Blob([response.data]));
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', 'weather_data.zip');
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(url);
      } catch (err) {
        this.error = 'Error downloading data';
        console.error('Download error:', err);
      } finally {
        this.downloading = false;
      }
    }
  },
  watch: {
    city(newVal) {
      if (newVal && this.startDate && this.endDate) {
        this.debouncedGetForecast();
      }
    },
    startDate(newVal) {
      if (this.city && newVal && this.endDate) {
        this.debouncedGetForecast();
      }
    },
    endDate(newVal) {
      if (this.city && this.startDate && newVal) {
        this.debouncedGetForecast();
      }
    }
  },
  created() {
    this.debouncedGetForecast = this.debounce(this.getForecast, 500);
  },
  mounted() {
    this.getForecast();
  }
};
</script>

<style>
.app {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

.forecast-form {
  display: flex;
  gap: 20px;
  margin-bottom: 20px;
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

input {
  padding: 8px;
  border: 1px solid #ccc;
  border-radius: 4px;
}

button {
  padding: 8px 16px;
  background-color: #4CAF50;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  align-self: flex-end;
}

button:disabled {
  background-color: #cccccc;
  cursor: not-allowed;
}

.error {
  color: red;
  margin-bottom: 20px;
}

.forecast-results {
  height: 400px;
  margin-top: 20px;
}

.city-select {
  padding: 8px;
  border: 1px solid #ccc;
  border-radius: 4px;
  background-color: white;
  min-width: 200px;
  cursor: pointer;
}

.city-select:focus {
  outline: none;
  border-color: #4CAF50;
  box-shadow: 0 0 0 2px rgba(74, 175, 80, 0.2);
}

/* Optional: Style the dropdown options */
.city-select option {
  padding: 8px;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.download-btn {
  background-color: #2196F3;
  color: white;
  padding: 8px 16px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
}

.download-btn:hover {
  background-color: #1976D2;
}

.download-btn:disabled {
  background-color: #BDBDBD;
  cursor: not-allowed;
}
</style> 