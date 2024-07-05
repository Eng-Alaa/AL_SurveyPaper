# Initialize the phone list
phone_list = []

def add_person():
    """Function to add a new person to the phone list"""
    name = input("Enter the name: ")
    address = input("Enter the address: ")
    phone1 = input("Enter the first phone number: ")
    phone2 = input("Enter the second phone number: ")
    person = [name, address, phone1, phone2]
    phone_list.append(person)
    print("Person added successfully!")

def update_person():
    """Function to update the details of a person in the phone list"""
    search_term = input("Enter the name, address, phone1, or phone2 to search for the person: ")
    found_person = None
    for person in phone_list:
        if (
            search_term.lower() in person[0].lower()
            or search_term.lower() in person[1].lower()
            or search_term in person[2]
            or search_term in person[3]
        ):
            found_person = person
            break
    if found_person:
        print("Found person:", found_person)
        new_name = input("Enter the new name (leave blank to keep the same): ")
        new_address = input("Enter the new address (leave blank to keep the same): ")
        new_phone1 = input("Enter the new first phone number (leave blank to keep the same): ")
        new_phone2 = input("Enter the new second phone number (leave blank to keep the same): ")
        if new_name:
            found_person[0] = new_name
        if new_address:
            found_person[1] = new_address
        if new_phone1:
            found_person[2] = new_phone1
        if new_phone2:
            found_person[3] = new_phone2
        print("Person updated successfully!")
    else:
        print("Person not found.")

def delete_person():
    """Function to delete a person from the phone list"""
    search_term = input("Enter the name, address, phone1, or phone2 to search for the person: ")
    found_person = None
    for person in phone_list:
        if (
            search_term.lower() in person[0].lower()
            or search_term.lower() in person[1].lower()
            or search_term in person[2]
            or search_term in person[3]
        ):
            found_person = person
            break
    if found_person:
        phone_list.remove(found_person)
        print("Person deleted successfully!")
    else:
        print("Person not found.")

def search_person():
    """Function to search for a person in the phone list"""
    search_term = input("Enter the name, address, phone1, or phone2 to search for the person: ")
    found_persons = []
    for person in phone_list:
        if (
            search_term.lower() in person[0].lower()
            or search_term.lower() in person[1].lower()
            or search_term in person[2]
            or search_term in person[3]
        ):
            found_persons.append(person)
    if found_persons:
        print("Found persons:")
        for person in found_persons:
            print(person)
    else:
        print("No person found.")

def display_all_persons():
    """Function to display all the persons in the phone list"""
    if not phone_list:
        print("The phone list is empty.")
    else:
        print("All persons in the phone list:")
        for person in phone_list:
            print("Name:", person[0])
            print("Address:", person[1])
            print("Phone1:", person[2])
            print("Phone2:", person[3])
            print("---")

while True:
    print("\nPhone List Menu:")
    print("1. Add a new person")
    print("2. Update a person")
    print("3. Delete a person")
    print("4. Search for a person")
    print("5. Display all persons")
    print("6. Exit")
    choice = input("Enter your choice (1-6): ")

    if choice == "1":
        add_person()
    elif choice == "2":
        update_person()
    elif choice == "3":
        delete_person()
    elif choice == "4":
        search_person()
    elif choice == "5":
        display_all_persons()
    elif choice == "6":
        print("Exiting the Phone List program.")
        break
    else:
        print("Invalid choice. Please try again.")