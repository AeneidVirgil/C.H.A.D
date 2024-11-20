import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import cmath
import random
import math

st.write("All expresion inputs have to be done in python language")

# Define the functions for each topic

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
    st.header("Thin Lens Equation")
    try:
        f = st.number_input("Enter focal length (f in cm)", value=0.0, format="%.2f")
        u = st.number_input("Enter object distance (u in cm)", value=0.0, format="%.2f")
        if f != 0 and u != 0:
            v = 1 / ((1 / f) - (1 / u))
            st.write(f"Image distance (v) = {v:.2f} cm")
        else:
            st.error("Focal length and object distance must be non-zero.")
    except:
        st.error("Please enter valid numerical inputs.")

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

# Main Streamlit App Function
def main():
    st.title("Comprehensive High-precision Algorithmic Device (C.H.A.D.)")

    # Create a dropdown menu for main topics
    tab = st.selectbox("Select a Topic", ["Calculus", "Physics"])

    # Handle Calculus options
    if tab == "Calculus":
        # Dropdown for Calculus operations
        operation = st.selectbox("Select an Operation", [
            "Limit Evaluation",
            "Derivative Evaluation",
            "Integral Evaluation",
            "Definite Integral Evaluation",
            "Series"
        ])

        if operation == "Limit Evaluation":
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
        # Dropdown for Physics topics
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

if __name__ == "__main__":
    main()
