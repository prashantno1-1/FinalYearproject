from flask import Flask, render_template, request, redirect, url_for, session,jsonify
import mysql.connector
import subprocess
import cv2
import mediapipe as mp
import numpy as np
import os

app = Flask(__name__)
app.secret_key = '1494f2fabc15b781ad271ea49f02ff5a'
gesture_process = None
volume_process = None 
mouse_process=None
bright_process=None

# MySQL Database Connection
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='root',
    database='gesture_control'
)
cursor = conn.cursor()

# Home Route
@app.route('/')
def home():
    return render_template('login.html')

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cursor.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
        user = cursor.fetchone()
        
        if user:
            session['user'] = username
            return redirect(url_for('dashboard'))
        else:
            return "Invalid credentials"
    return render_template('login.html')

# Register Route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
        conn.commit()
        return redirect(url_for('login'))
    return render_template('register.html')

# Dashboard Route
@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', user=session['user'])

# Gesture Recognition Route
@app.route('/start_gestures')
def start_gestures():
    if 'user' not in session:
        return "Unauthorized Access"
    os.system('python gestures/gesture_control.py')
    return redirect(url_for('dashboard'))

# Logout Route
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

 # ✅ Gesture Control Activation/Deactivation
@app.route('/gesture_control')
def gesture_control():
    global gesture_process
    global volume_process
    global mouse_process
    global bright_process
    action = request.args.get('action')

    if action == 'activate':
        # Start the gesture control process if it's not already running
        if gesture_process is None or gesture_process.poll() is not None:
            gesture_process = subprocess.Popen(['python', 'gesture_control.py'])
            return jsonify({"message": "Gesture control activated!"})
        else:
            return jsonify({"message": "Gesture control is already active!"})

    elif action == 'deactivate':
        # Terminate the gesture control process if it's running
        if gesture_process is not None:
            gesture_process.terminate()  # Terminate the process
            gesture_process = None
            return jsonify({"message": "Gesture control deactivated!"})
        else:
            return jsonify({"message": "No active gesture control to deactivate!"})
        
    elif action == 'start_volume':
        if volume_process is None or volume_process.poll() is not None:
            # Replace with actual volume control logic
            volume_process = subprocess.Popen(['python', 'mute.py'])  # Example volume control script
            return jsonify({"message": "mute/unmute control started!"})
        else:
            return jsonify({"message": "Volume control is already running!"})

    elif action == 'stop_volume':
        if volume_process is not None:
            volume_process.terminate()  # Terminate volume control process
            volume_process = None
            return jsonify({"message": "mute/unmute control stopped!"})
        else:
            return jsonify({"message": "No active volume control to stop!"})
        
    elif action == 'start_mouse':
        if mouse_process is None or mouse_process.poll() is not None:
            # Replace with actual volume control logic
            mouse_process = subprocess.Popen(['python', 'vmouse.py'])  # Example volume control script
            return jsonify({"message": "mouse control started!"})
        else:
            return jsonify({"message": "mouse control is already running!"})

    elif action == 'stop_mouse':
        if mouse_process is not None:
            mouse_process.terminate()  # Terminate volume control process
            mouse_process = None
            return jsonify({"message": "mouse control stopped!"})
        else:
            return jsonify({"message": "No active mouse control to stop!"})
        
    elif action == 'start_bright':
        if bright_process is None or bright_process.poll() is not None:
            # Replace with actual volume control logic
            bright_process = subprocess.Popen(['python', 'brightness.py'])  # Example volume control script
            return jsonify({"message": "brightness control started!"})
        else:
            return jsonify({"message": "brightness control is already running!"})

    elif action == 'stop_bright':
        if bright_process is not None:
            bright_process.terminate()  # Terminate volume control process
            bright_process = None
            return jsonify({"message": "brightness control stopped!"})
        else:
            return jsonify({"message": "No active brightness control to stop!"})

    return jsonify({"message": "Invalid action!"})

if __name__ == '__main__':
    app.run(debug=True, port=9000)
