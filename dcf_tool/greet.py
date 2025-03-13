from datetime import datetime

def greet():
    """Returns a greeting based on the time of day."""
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning!"
    elif hour < 18:
        return "Good afternoon!"
    else:
        return "Good evening!"

def main():
    while True:
        print("\n" + greet())
        user_input = input("Type 'exit' to quit or press Enter to get another greeting: ").strip().lower()
        if user_input == "exit":
            print("Goodbye!")
            break

if __name__ == "__main__":
    main()