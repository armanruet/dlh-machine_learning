-- Create script that computes the average temp by city
SELECT city, AVG(temperature) as avg_temp FROM temperatures GROUP BY city ORDER BY avg_temp DESC
