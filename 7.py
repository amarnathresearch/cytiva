import mysql.connector

# Connect to MySQL database
try:
    connection = mysql.connector.connect(
        host='localhost',       # Replace with your MySQL host
        user='root',            # Replace with your MySQL username
        password='xyz',  # Replace with your MySQL password
        database='cytiva'   # Replace with your database name
    )

    if connection.is_connected():
        print("Connected to MySQL database")

        # Create a cursor object
        cursor = connection.cursor()

        # Execute a query to select data from 'user' table
        cursor.execute("SELECT * FROM user")

        # Fetch all rows
        rows = cursor.fetchall()

        # Print the data
        print("Data from 'user' table:")
        for row in rows:
            print(row)

        # Close the cursor and connection
        cursor.close()
        connection.close()
        print("Connection closed")

except mysql.connector.Error as e:
    print(f"Error connecting to MySQL: {e}")
