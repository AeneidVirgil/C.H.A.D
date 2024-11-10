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

import sympy as sp
import streamlit as st

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



# Streamlit app code
def main():
    st.title("Comprehensive High-precision Algorithmic Device (C.H.A.D.)")

    # Create a dropdown menu for sections
    tab = st.selectbox("Select a Topic", ["Calculus", "Other"])

    if tab == "Calculus":
        # Create another dropdown for the specific operations in Calculus 1
        operation = st.selectbox("Select an Operation", ["Limit Evaluation", "Derivative Evaluation", "Integral Evaluation", "Definite Integral Evaluation", "Series"])

        if operation == "Limit Evaluation":
            step_by_step_limit_evaluator()
        elif operation == "Derivative Evaluation":
            step_by_step_derivative_evaluator()
        elif operation == "Integral Evaluation":
            step_by_step_integral_evaluator()
        elif operation == "Definite Integral Evaluation":
            step_by_step_definite_integral_evaluator()
        elif operation == "Series Evaluation":
          step_by_step_series_evaluator()
      
    else:
        st.header("Other Topics")
        st.write("This section is for other topics. Add more content as needed.")

if __name__ == "__main__":
    main()



