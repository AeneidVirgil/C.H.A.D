import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import cmath
import random
import math
from PIL import Image
import pytesseract
from streamlit_drawable_canvas import st_canvas


st.write("All expresion inputs have to be done in python language")

# Define the functions for each topic
#Calulus 
# Function to evaluate the limit step by step
def step_by_step_limit_evaluator():
    # Define the variable
    x = sp.symbols('x')

    # Streamlit input for the mathematical expression with a unique key
    expr_input = st.text_input("Enter the expression to evaluate the limit (in terms of x):", "sin(x)/x", key="limit_expression_input")

    if expr_input:
        # Parse the input expression
        try:
            expr = sp.sympify(expr_input)
        except sp.SympifyError:
            st.error("Invalid expression. Please try again.")
            return

        # Streamlit input for the limit point
        point_input = st.text_input("Enter the point at which to evaluate the limit (e.g., 0, infinity, -infinity):", "0", key="limit_point_input")

        if point_input:
            # Parse the limit point
            try:
                if point_input == "infinity":
                    point = sp.oo
                elif point_input == "-infinity":
                    point = -sp.oo
                else:
                    point = sp.sympify(point_input)
            except ValueError:
                st.error("Invalid point. Please try again.")
                return

            # Display the expression and point
            st.write(f"\nYou entered: {expr} at x = {point}")

            # Step 1: Direct substitution
            direct_sub = expr.subs(x, point)
            if direct_sub.is_finite:
                st.write(f"Step 1: Direct substitution gives: {direct_sub}")
            else:
                st.write(f"Step 1: Direct substitution gives an indeterminate form: {direct_sub}")

            # Step 2: Simplify the expression
            simplified_expr = sp.simplify(expr)
            st.write(f"Step 2: Simplified expression: {simplified_expr}")

            # Step 3: Check the limit
            limit_result = sp.limit(expr, x, point)
            st.write(f"Step 3: The limit is: {limit_result}")


# Function to evaluate the derivative step by step
def step_by_step_derivative_evaluator():
    # Define the variable
    x = sp.symbols('x')

    # Streamlit input for the mathematical expression with a unique key
    expr_input = st.text_input("Enter the expression to differentiate (in terms of x):", "x**2", key="derivative_expression_input")

    if expr_input:
        # Parse the input expression
        try:
            expr = sp.sympify(expr_input)
        except sp.SympifyError:
            st.error("Invalid expression. Please try again.")
            return

        # Display the expression
        st.write(f"\nYou entered: {expr}\n")

        # Step 1: Display the expression to differentiate
        st.write(f"Step 1: We will differentiate the expression: d/dx({expr})")

        # Step 2: Perform the differentiation
        derivative_result = sp.diff(expr, x)

        # Step 3: Show the result
        st.write(f"\nStep 2: The derivative is: d/dx({expr}) = {derivative_result}")


# Function to evaluate the integral step by step
def step_by_step_integral_evaluator():
    # Define the variable
    x = sp.symbols('x')

    # Streamlit input for the mathematical expression with a unique key
    expr_input = st.text_input("Enter the expression to integrate (in terms of x):", "x**2", key="integral_expression_input")

    if expr_input:
        # Parse the input expression
        try:
            expr = sp.sympify(expr_input)
        except sp.SympifyError:
            st.error("Invalid expression. Please try again.")
            return

        # Step 1: Display the expression to integrate
        st.write(f"Step 1: We will integrate the expression: ∫{expr} dx")

        # Step 2: Perform the integration
        integral_result = sp.integrate(expr, x)

        # Step 3: Show the result
        st.write(f"\nStep 2: The integral is: ∫{expr} dx = {integral_result} + C")


# Function to evaluate the definite integral step by step
def step_by_step_definite_integral_evaluator():
    # Define the variable
    x = sp.symbols('x')

    # Streamlit input for the mathematical expression with a unique key
    expr_input = st.text_input("Enter the expression to integrate (in terms of x):", "x**2", key="definite_integral_expression_input")

    if expr_input:
        # Parse the input expression
        try:
            expr = sp.sympify(expr_input)
        except sp.SympifyError:
            st.error("Invalid expression. Please try again.")
            return

        # Streamlit input for the limits of integration with unique keys
        a_input = st.number_input("Enter the lower limit of integration:", value=0.0, key="definite_lower_limit")
        b_input = st.number_input("Enter the upper limit of integration:", value=1.0, key="definite_upper_limit")

        if a_input != b_input:
            # Display the expression
            st.write(f"\nYou entered: {expr}\n")

            # Step 1: Display the expression to integrate
            st.write(f"Step 1: We will integrate the expression: ∫ from {a_input} to {b_input} of {expr} dx")

            # Step 2: Perform the definite integration
            integral_result = sp.integrate(expr, (x, a_input, b_input))

            # Step 3: Show the result
            st.write(f"\nStep 2: The definite integral is: ∫ from {a_input} to {b_input} of {expr} dx = {integral_result}")

            # Step 4: Optional - Display differentiation to verify
            verification = sp.diff(sp.integrate(expr, x), x)
            st.write(f"\nStep 3: Verifying by differentiating the indefinite integral: d/dx(∫{expr} dx) = {verification}")
        else:
            st.warning("The lower limit and upper limit must be different for the integration to be valid.")

#Series


def step_by_step_series_evaluator():
    n = sp.symbols('n')

    expr_input = st.text_input("Enter the expression for the series (in terms of n): ")

    try:
        expr = sp.sympify(expr_input)
    except sp.SympifyError:
        st.write("Invalid expression. Please enter a valid mathematical expression.")
        return  # this return is fine, as it's inside the function

    # Ask if the user wants to evaluate a finite series
    finite_input = st.selectbox("Do you want to evaluate a finite series?", ['yes', 'no'])

    if finite_input == 'yes':
        a_input = st.text_input("Enter the lower limit of summation (start value): ")
        b_input = st.text_input("Enter the upper limit of summation (end value): ")

        try:
            a = int(a_input)
            b = int(b_input)
        except ValueError:
            st.write("Invalid limits. Please enter integer values.")
            return

        # Perform the summation
        series_result = sp.Sum(expr, (n, a, b)).doit()

        st.write(f"The result of the finite series is: Σ from {a} to {b} of {expr} = {series_result}")

    elif finite_input == 'no':
        # Perform the summation for the infinite series
        series_result = sp.Sum(expr, (n, 1, sp.oo)).doit()

        st.write(f"The result of the infinite series is: Σ from n=1 to ∞ of {expr} = {series_result}")

    else:
        st.write("Invalid option. Please enter 'yes' or 'no'.")
        return  # This return is also inside the function

#Taylor Series

#Code for recognizing image 
def recognize_math_expression(image):
    try:
        text = pytesseract.image_to_string(image, config="--psm 6")
        return text.strip()
    except Exception as e:
        st.error(f"Error recognizing the equation: {e}")
        return None

# Modified Limit Evaluator with Math Equation Reader
def step_by_step_limit_evaluator_with_reader():
    st.header("Step-by-Step Limit Evaluator with Handwritten Input")

    # Option to upload an image or draw an equation
    st.write("Upload an image of your handwritten equation or draw it below:")
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    with col2:
        canvas_result = st_canvas(
            stroke_width=2,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=150,
            width=300,
            drawing_mode="freedraw",
            key="canvas",
        )

    # Process input
    if uploaded_file or (canvas_result and canvas_result.image_data is not None):
        image = Image.open(uploaded_file) if uploaded_file else Image.fromarray(canvas_result.image_data.astype("uint8"))
        st.image(image, caption="Uploaded or Drawn Image", use_column_width=True)

        # Recognize the math expression
        recognized_expr = recognize_math_expression(image)
        if recognized_expr:
            st.write(f"Recognized Expression: `{recognized_expr}`")

            # Try to parse the recognized expression into sympy
            try:
                x = sp.symbols('x')  # Define the variable
                expr = sp.sympify(recognized_expr)
                st.write(f"Parsed Expression: {expr}")

                # Continue with limit evaluation
                point_input = st.text_input("Enter the limit point (e.g., 0, infinity, -infinity):", "0")
                if point_input:
                    point = sp.sympify(point_input) if point_input not in ["infinity", "-infinity"] else sp.oo if point_input == "infinity" else -sp.oo
                    limit_result = sp.limit(expr, x, point)
                    st.write(f"The limit is: {limit_result}")
            except sp.SympifyError:
                st.error("Unable to parse the recognized expression. Please try again.")

#PHYSICS 
#Kinematics 
def kinematics_equations():
    st.header("Kinematics Calculator")
    st.write("Choose a variable to solve for:")

    # Choice selection using radio buttons
    choice = st.radio("Select what you want to solve for:", ["Final velocity (v)", "Displacement (x)", "Time (t)"])

    if choice == "Final velocity (v)":
        # Solve for final velocity: v = u + at
        u = st.number_input("Enter initial velocity (u) in m/s:", value=0.0, format="%.2f")
        a = st.number_input("Enter acceleration (a) in m/s^2:", value=0.0, format="%.2f")
        t = st.number_input("Enter time (t) in seconds:", value=0.0, format="%.2f")
        
        v = u + a * t
        st.write(f"Final velocity (v) = {v:.2f} m/s")

    elif choice == "Displacement (x)":
        # Solve for displacement: x = ut + 0.5 * a * t^2
        u = st.number_input("Enter initial velocity (u) in m/s:", value=0.0, format="%.2f")
        a = st.number_input("Enter acceleration (a) in m/s^2:", value=0.0, format="%.2f")
        t = st.number_input("Enter time (t) in seconds:", value=0.0, format="%.2f")
        
        x = u * t + 0.5 * a * t**2
        st.write(f"Displacement (x) = {x:.2f} meters")

    elif choice == "Time (t)":
        # Solve for time: t = (v - u) / a
        u = st.number_input("Enter initial velocity (u) in m/s:", value=0.0, format="%.2f")
        v = st.number_input("Enter final velocity (v) in m/s:", value=0.0, format="%.2f")
        a = st.number_input("Enter acceleration (a) in m/s^2:", value=0.0, format="%.2f")
        
        if a != 0:
            t = (v - u) / a
            st.write(f"Time (t) = {t:.2f} seconds")
        else:
            st.error("Acceleration cannot be zero!")

# Projectile Motion with Graph
def projectile_motion():
    st.header("Projectile Motion")
    v0 = st.number_input("Enter initial velocity (v0) in m/s:", value=10.0, format="%.2f")
    angle_deg = st.number_input("Enter launch angle in degrees:", value=45.0, format="%.2f")
    mass = st.number_input("Enter the mass of the object in kg (optional, does not affect trajectory):", value=1.0, format="%.2f")
    start_height = st.number_input("Enter starting height in meters:", value=0.0, format="%.2f")

    angle_rad = np.radians(angle_deg)
    g = 9.81
    v0_x = v0 * np.cos(angle_rad)
    v0_y = v0 * np.sin(angle_rad)
    discriminant = v0_y**2 + 2 * g * start_height

    if discriminant < 0:
        st.error("The given parameters will not allow the projectile to hit the ground.")
        return

    t_flight = (v0_y + np.sqrt(discriminant)) / g
    t = np.linspace(0, t_flight, num=500)
    x = v0_x * t
    y = start_height + v0_y * t - 0.5 * g * t**2

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, label=f"Angle: {angle_deg}°, Velocity: {v0} m/s")
    ax.set_title("Projectile Motion Path")
    ax.set_xlabel("Horizontal Distance (m)")
    ax.set_ylabel("Vertical Height (m)")
    ax.legend()
    ax.grid(True)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)

    st.pyplot(fig)

# Vector Operations 
def get_vector_input(vector_name):
    components = st.text_input(f"Enter the components of vector {vector_name} separated by spaces:")
    if components.strip(): 
        try:
            return np.array([float(component) for component in components.split()])
        except ValueError:
            st.error(f"Invalid input for vector {vector_name}. Please enter numbers separated by spaces.")
            return None
    return None

def vector_operations_tab():
    st.header("Vector Operations")

    # Get user inputs for vectors
    a = get_vector_input("a")
    b = get_vector_input("b")

    if a is not None and b is not None:
        if len(a) == len(b):  
            # Dot product
            dot_product = np.dot(a, b)
            st.write(f"Dot product: {dot_product}")

            # Cross product (only for 3D vectors)
            if len(a) == 3:
                cross_product = np.cross(a, b)
                st.write(f"Cross product: {cross_product}")

                # Magnitude of cross product
                magnitude_cross_product = np.linalg.norm(cross_product)
                st.write(f"Magnitude of cross product: {magnitude_cross_product:.2f}")
            else:
                st.write("Cross product is only defined for 3D vectors.")

            # Magnitudes of vectors
            magnitude_a = np.linalg.norm(a)
            st.write(f"Magnitude of a: {magnitude_a:.2f}")

            magnitude_b = np.linalg.norm(b)
            st.write(f"Magnitude of b: {magnitude_b:.2f}")
        else:
            st.error("Vectors must have the same dimension.")
    else:
        st.write("Please enter valid vector components for both vectors.")


#Snells Law
def snells_law():
    st.header("Snell's Law")
    try:
        n1 = st.number_input("Enter refractive index of medium 1 (n1)", value=1.0, format="%.2f")
        n2 = st.number_input("Enter refractive index of medium 2 (n2)", value=1.0, format="%.2f")
        theta1 = st.number_input("Enter angle of incidence (in degrees)", value=0.0, format="%.2f")
        theta1_rad = math.radians(theta1)
        if (n1 / n2) * math.sin(theta1_rad) <= 1:
            theta2_rad = math.asin((n1 / n2) * math.sin(theta1_rad))
            theta2 = math.degrees(theta2_rad)
            st.write(f"Angle of refraction (θ2) = {theta2:.2f} degrees")
        else:
            st.error("Total internal reflection occurred.")
    except:
        st.error("Invalid input or total internal reflection occurred.")

def thin_lens_equation():
    st.header("Thin Lens Equation Solver")

    try:
        s = st.number_input("Enter object distance (s in cm)", value=0.0, format="%.2f")
        s_prime = st.number_input("Enter image distance (s' in cm)", value=0.0, format="%.2f")

        # Check if both s and s' are non-zero
        if s != 0 and s_prime != 0:
            f = 1 / (1 / s + 1 / s_prime)  # Formula for the focal length
            st.write(f"Focal length (f) = {f:.2f} cm")
        else:
            st.error("Object distance (s) and image distance (s') must be non-zero.")
    except Exception as e:
        st.error(f"Error: {e}. Please enter valid numerical inputs.")

def lens_maker_equation():
    st.header("Lens Maker's Equation")
    try:
        n = st.number_input("Enter refractive index of the lens (n)", value=1.5, format="%.2f")
        r1 = st.number_input("Enter radius of curvature of first surface (R1 in cm)", value=0.0, format="%.2f")
        r2 = st.number_input("Enter radius of curvature of second surface (R2 in cm)", value=0.0, format="%.2f")
        if r1 != 0 and r2 != 0:
            f = 1 / ((n - 1) * ((1 / r1) - (1 / r2)))
            st.write(f"Focal length (f) = {f:.2f} cm")
        else:
            st.error("Radii of curvature must be non-zero.")
    except:
        st.error("Please enter valid numerical inputs.")

#Statistics 

def plot_histogram(data, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=15, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    st.pyplot(plt)

def plot_scatter(x, y, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    st.pyplot(plt)

# Function to calculate binomial probability
def binomial_probability(n, k, p):
    from math import comb
    return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

# Function to calculate normal probability density
def normal_pdf(x, mu, sigma):
    return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Function to calculate basic probability
def basic_probability(event_outcomes, total_outcomes):
    return event_outcomes / total_outcomes

# Function to calculate conditional probability
def conditional_probability(event_a_and_b, event_b):
    return event_a_and_b / event_b

# Function to check independence
def check_independence(prob_a, prob_b, prob_a_and_b):
    return prob_a_and_b == prob_a * prob_b

# Function to generate random numbers
def generate_random_numbers(count, start, end):
    return [random.randint(start, end) for _ in range(count)]

# Function to calculate mean, median, and mode
def calculate_stats(data):
    mean = statistics.mean(data)
    median = statistics.median(data)
    try:
        mode = statistics.mode(data)
    except statistics.StatisticsError:
        mode = "No unique mode found"
    return mean, median, mode

# Function to calculate range, variance, standard deviation, IQR, and identify outliers
def calculate_advanced_stats(data):
    data_range = max(data) - min(data)
    variance = statistics.variance(data)
    std_dev = statistics.stdev(data)
    sorted_data = sorted(data)
    q1 = sorted_data[len(data) // 4]
    q3 = sorted_data[(len(data) * 3) // 4]
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = [x for x in data if x < lower_bound or x > upper_bound]
    return data_range, variance, std_dev, iqr, outliers

# Function to generate sampling distribution of the sample mean
def sampling_distribution(population, sample_size, num_samples):
    sample_means = []
    for _ in range(num_samples):
        sample = random.sample(population, sample_size)
        sample_means.append(statistics.mean(sample))
    return sample_means

# Function to perform simple random sampling
def simple_random_sampling(data, sample_size):
    if sample_size > len(data) or sample_size < 0:
        raise ValueError("Sample size must be between 0 and the size of the population.")
    return random.sample(data, sample_size)

# Function to calculate Chi-Square statistic
def chi_square_statistic(observed, expected):
    chi2 = 0
    for o, e in zip(observed, expected):
        chi2 += (o - e) ** 2 / e
    return chi2

# Function to calculate confidence interval
def confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = statistics.mean(data)
    std_err = statistics.stdev(data) / math.sqrt(n)
    margin_of_error = std_err * 1.96  # For 95% confidence
    return mean - margin_of_error, mean + margin_of_error

def confidence_interval_calculations():
    data_input = st.text_input("Enter your data set, separated by commas (e.g., 10, 12, 23, 23, 16, 23, 21, 16, 18, 20):", "10,12,23,23,16,23,21,16,18,20")
    data = list(map(int, data_input.split(',')))
    conf_int = confidence_interval(data)
    st.write(f"95% Confidence Interval: {conf_int}")

# Function to perform stratified random sampling
def stratified_random_sampling(data, strata, sample_size):
    stratified_sample = []
    for stratum in strata:
        stratum_data = [item for item in data if item[1] == stratum]
        stratum_sample_size = int(sample_size * (len(stratum_data) / len(data)))
        stratified_sample.extend(random.sample(stratum_data, stratum_sample_size))
    return stratified_sample

def stratified_random_sampling_calculations():
    data_input = st.text_input("Enter your data set as value,stratum pairs, separated by commas (e.g., 1,A,2,A,3,B): ", "1,A,2,A,3,B")
    strata_input = st.text_input("Enter your strata, separated by commas (e.g., A,B,C): ", "A,B,C")
    sample_size = st.number_input("Enter the sample size: ", min_value=1, value=1)

    data = [(int(data_input.split(',')[i]), data_input.split(',')[i+1]) for i in range(0, len(data_input.split(',')), 2)]
    strata = strata_input.split(',')

    try:
        sample = stratified_random_sampling(data, strata, sample_size)
        st.write(f"Stratified Random Sample: {sample}")
    except ValueError as e:
        st.error(e)

# Function to calculate expected frequencies
def calculate_expected(observed):
    row_totals = [sum(row) for row in observed]
    col_totals = [sum(col) for col in zip(*observed)]
    total = sum(row_totals)
    expected = [[(row_total * col_total) / total for col_total in col_totals] for row_total in row_totals]
    return expected

# Function to flip a coin multiple times
def flip_coin(times):
    heads = 0
    tails = 0
    for _ in range(times):
        result = random.choice(['Heads', 'Tails'])
        if result == 'Heads':
            heads += 1
        else:
            tails += 1
    return heads, tails

def distribution_calculations():
    dist_type = st.selectbox("Which distribution would you like to calculate?", ("binomial", "normal"))
    
    if dist_type == 'binomial':
        n = st.number_input("Enter the number of trials", min_value=1, value=10)
        p = st.number_input("Enter the probability of success", min_value=0.0, max_value=1.0, value=0.5)
        x = list(range(n + 1))
        binom_pmf = [binomial_probability(n, k, p) for k in x]
        for k, prob in zip(x, binom_pmf):
            st.write(f"Number of successes: {k}, Probability: {prob}")
    elif dist_type == 'normal':
        mu = st.number_input("Enter the mean", value=0.0)
        sigma = st.number_input("Enter the standard deviation", min_value=0.0, value=1.0)
        x = [mu + sigma * i / 10 for i in range(-30, 31)]
        norm_pdf = [normal_pdf(val, mu, sigma) for val in x]
        for val, prob in zip(x, norm_pdf):
            st.write(f"Value: {val}, Probability Density: {prob}")

def probability_calculations():
    num_events = st.number_input("How many events are there?", min_value=1, value=1)
    event_names = []
    event_outcomes = []

    for i in range(num_events):
        event_name = st.text_input(f"Enter the name of event {i+1}", f"Event {i+1}")
        outcomes = st.number_input(f"Enter the number of outcomes for {event_name}", min_value=1, value=30)
        event_names.append(event_name)
        event_outcomes.append(outcomes)

    total_outcomes = st.number_input("Enter the total number of outcomes", min_value=1, value=100)
    probabilities = {name: basic_probability(outcomes, total_outcomes) for name, outcomes in zip(event_names, event_outcomes)}

    st.write("Probabilities:")
    for name, prob in probabilities.items():
        st.write(f"Probability of {name}: {prob}")

def random_number_generation():
    start = st.number_input("Enter the lower range", value=1)
    end = st.number_input("Enter the upper range", value=100)
    count = st.number_input("How many random numbers do you want to generate?", min_value=1, value=10)
    random_numbers = generate_random_numbers(count, start, end)
    st.write(f"Random Numbers: {random_numbers}")

def statistics_calculations():
    data_input = st.text_input("Enter your data set, separated by commas", "10,20,30")
    data = list(map(int, data_input.split(',')))
    mean, median, mode = calculate_stats(data)
    st.write(f"Mean: {mean}")
    st.write(f"Median: {median}")
    st.write(f"Mode: {mode}")
    plot_histogram(data, 'Histogram of Data', 'Value', 'Frequency')

def advanced_statistics_calculations():
    data_input = st.text_input("Enter your data set, separated by commas", "1,2,3")
    data = list(map(int, data_input.split(',')))
    data_range, variance, std_dev, iqr, outliers = calculate_advanced_stats(data)
    st.write(f"Range: {data_range}")
    st.write(f"Variance: {variance}")
    st.write(f"Standard Deviation: {std_dev}")
    st.write(f"Interquartile Range (IQR): {iqr}")
    st.write(f"Outliers: {outliers}")
    plot_histogram(data, 'Histogram of Data', 'Value', 'Frequency')

def sampling_distribution_calculations():
    population_size = st.number_input("Enter the population size for sampling distribution", min_value=1, value=100)
    population = list(range(1, population_size + 1))
    sample_size = st.number_input("Enter the sample size for sampling distribution", min_value=1, value=10)
    num_samples = st.number_input("Enter the number of samples", min_value=1, value=100)
    sample_means = sampling_distribution(population, sample_size, num_samples)
    st.write(f"Sample Means: {sample_means}")
    st.write(f"Mean of Sample Means: {statistics.mean(sample_means)}")
    st.write(f"Standard Deviation of Sample Means: {statistics.stdev(sample_means)}")
    st.write("Shape: According to the Central Limit Theorem, the sampling distribution of the sample mean will be approximately normally distributed if the sample size is sufficiently large.")
    plot_histogram(sample_means, 'Sampling Distribution of Sample Means', 'Sample Mean', 'Frequency')

def simple_random_sampling_calculations():
    population_size = st.number_input("Enter the population size for random sampling", min_value=1, value=100)
    data = list(range(1, population_size + 1))
    sample_size = st.number_input("Enter the sample size for random sampling", min_value=1, value=10)
    try:
        sample = simple_random_sampling(data, sample_size)
        st.write(f"Random Sample: {sample}")
    except ValueError as e:
        st.error(e)

def chi_square_test():
    num_rows = st.number_input("Enter the number of rows for Chi-Square test", min_value=1, value=2)
    st.write("Enter the observed data row by row, with values separated by commas (e.g., 10,20,30):")
    observed = []
    for i in range(num_rows):
        row = list(map(int, st.text_input(f"Row {i+1}").split(',')))
        observed.append(row)
    
    expected = calculate_expected(observed)
    chi2 = chi_square_statistic([item for sublist in observed for item in sublist], [item for sublist in expected for item in sublist])
    dof = (num_rows - 1) * (len(observed[0]) - 1)

    st.write(f"Chi-Square Statistic: {chi2}")
    st.write(f"Degrees of Freedom: {dof}")
    st.write("Expected Frequencies:")
    for row in expected:
        st.write(row)

def coin_flip_simulation():
    times = st.number_input("How many times do you want to flip the coin?", min_value=1, value=10)
    heads, tails = flip_coin(times)
    st.write(f"Heads: {heads}")
    st.write(f"Tails: {tails}")


#Streamlit Code
def main():
    st.title("Comprehensive High-precision Algorithmic Device (C.H.A.D.)")

    # Create a dropdown menu for main topics
    tab = st.selectbox("Select a Topic", ["Calculus", "Physics", "Statistics"])

    # Handle Calculus options
    if tab == "Calculus":
        operation = st.selectbox("Select an Operation", [
            "Limit Evaluation (with Reader)",
            "Derivative Evaluation",
            "Integral Evaluation",
            "Definite Integral Evaluation",
            "Series"
        ])

        if operation == "Limit Evaluation (with Reader)":
            step_by_step_limit_evaluator()
        elif operation == "Derivative Evaluation":
            step_by_step_derivative_evaluator()
        elif operation == "Integral Evaluation":
            step_by_step_integral_evaluator()
        elif operation == "Definite Integral Evaluation":
            step_by_step_definite_integral_evaluator()
        elif operation == "Series":
            step_by_step_series_evaluator()

    # Handle Physics options
    elif tab == "Physics":
        physics_topic = st.selectbox("Select a Physics Topic", [
            "Kinematics",
            "Projectile Motion",
            "Vector Operations",
            "Snell's Law",
            "Thin Lens Equation",
            "Lens Maker's Equation"
        ])

        if physics_topic == "Kinematics":
            kinematics_equations()
        elif physics_topic == "Projectile Motion":
            projectile_motion()
        elif physics_topic == "Vector Operations":
            vector_operations_tab()
        elif physics_topic == "Snell's Law":
            snells_law()
        elif physics_topic == "Thin Lens Equation":
            thin_lens_equation()
        elif physics_topic == "Lens Maker's Equation":
            lens_maker_equation()

    # Handle Statistics options
    elif tab == "Statistics":
        # Sidebar for Statistical Calculations
        st.sidebar.header("Choose a calculation")
        choice = st.sidebar.selectbox("Select one", [
            "Distributions (binomial/normal)", 
            "Probabilities", 
            "Random Number Generation", 
            "Statistics Calculations", 
            "Advanced Statistics Calculations", 
            "Sampling Distribution Calculations", 
            "Simple Random Sampling", 
            "Stratified Random Sampling", 
            "Chi-Square Test", 
            "Coin Flip Simulation", 
            "Confidence Interval"
        ])

        if choice == "Distributions (binomial/normal)":
            distribution_calculations()
        elif choice == "Probabilities":
            probability_calculations()
        elif choice == "Random Number Generation":
            random_number_generation()
        elif choice == "Statistics Calculations":
            statistics_calculations()
        elif choice == "Advanced Statistics Calculations":
            advanced_statistics_calculations()
        elif choice == "Sampling Distribution Calculations":
            sampling_distribution_calculations()
        elif choice == "Simple Random Sampling":
            simple_random_sampling_calculations()
        elif choice == "Stratified Random Sampling":
            stratified_random_sampling_calculations()
        elif choice == "Chi-Square Test":
            chi_square_test()
        elif choice == "Coin Flip Simulation":
            coin_flip_simulation()
        elif choice == "Confidence Interval":
            confidence_interval_calculations()

if __name__ == "__main__":
    main()
