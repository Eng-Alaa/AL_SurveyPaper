import pandas as pd
import os

# Initialize the phone list
phone_list = []

def add_person():
    """Function to add a new person to the phone list"""
    name = input("Enter the name: ")
    address = input("Enter the address: ")
    phone1 = input("Enter the first phone number: ")
    phone2 = input("Enter the second phone number: ")
    person = {
        "name": name,
        "address": address,
        "phone1": phone1,
        "phone2": phone2
    }
    phone_list.append(person)
    # Check if phone_list.csv exists, if not, create a new file
    if not os.path.isfile('phone_list.csv'):
        phone_df = pd.DataFrame(phone_list)
        phone_df.to_csv('phone_list.csv', index=False)
        print("Person added successfully!")
    # If the file exists, append the new data to it
    else:
        phone_df = pd.DataFrame(phone_list)
        phone_df.to_csv('phone_list.csv', mode='a', header=False, index=False)
        print("Person added successfully!")


def update_person():
    """Function to update the details of a person in the phone list and save to CSV"""
    phone_list = pd.read_csv('phone_list.csv', header=None, names=['name', 'address', 'phone1', 'phone2']).to_dict('records')
    search_term = input("Enter the name, address, phone1, or phone2 to search for the person: ")
    phone_df = pd.DataFrame(phone_list)
    found_person = phone_df[(phone_df['name']==search_term) |
                           (phone_df['address']==search_term) |
                           (phone_df['phone1'] == search_term) |
                           (phone_df['phone2'] == search_term)]

    if not found_person.empty:
        print("Found person:", found_person.iloc[0].to_dict())
        new_name = input("Enter the new name (leave blank to keep the same): ")
        new_address = input("Enter the new address (leave blank to keep the same): ")
        new_phone1 = input("Enter the new first phone number (leave blank to keep the same): ")
        new_phone2 = input("Enter the new second phone number (leave blank to keep the same): ")

        # Update the found person's information
        # Update the found person's information
        if new_name:
            found_person.at[found_person.index[0], 'name'] = new_name
        if new_address:
            found_person.at[found_person.index[0], 'address'] = new_address
        if new_phone1:
            found_person.at[found_person.index[0], 'phone1'] = new_phone1
        if new_phone2:
            found_person.at[found_person.index[0], 'phone2'] = new_phone2

        # Update the DataFrame and write to CSV
        phone_df.update(found_person)


        # Update the DataFrame and write to CSV
        phone_df = pd.DataFrame(phone_list)
        phone_df.to_csv('phone_list.csv', index=False, header=False)
        print("Person updated successfully and saved to phone_list.csv!")
    else:
        print("Person not found.")

def delete_person():
    """Function to delete a person from the phone list"""
    phone_list = pd.read_csv('phone_list.csv').to_dict('records')
    search_term = input("Enter the name, address, phone1, or phone2 to search for the person: ")
    found_person = None
    for person in phone_list:
        if (
            search_term.lower() in person["name"].lower()
            or search_term.lower() in person["address"].lower()
            or search_term in person["phone1"]
            or search_term in person["phone2"]
        ):
            found_person = person
            break
    if found_person:
        phone_list.remove(found_person)
        # Update the DataFrame and write to CSV
        phone_df = pd.DataFrame(phone_list)
        phone_df.to_csv('phone_list.csv', index=False)
        print("Person deleted successfully!")
    else:
        print("Person not found.")

def search_person():
    """Function to search for a person in the phone list"""
    phone_list = pd.read_csv('phone_list.csv').to_dict('records')
    search_term = input("Enter the name, address, phone1, or phone2 to search for the person: ")
    phone_df = pd.DataFrame(phone_list)
    found_persons = phone_df[(phone_df['name'].str.contains(search_term, case=False)) |
                            (phone_df['address'].str.contains(search_term, case=False)) |
                            (phone_df['phone1'] == search_term) |
                            (phone_df['phone2'] == search_term)]

    if not found_persons.empty:
        print("Found persons:")
        print(found_persons.to_string(index=False))
    else:
        print("No person found.")

def display_all_persons():
    """Function to display all the persons in the phone list"""
    phone_list = pd.read_csv('phone_list.csv').to_dict('records')

    phone_df = pd.DataFrame(phone_list)
    if phone_df.empty:
        print("The phone list is empty.")
    else:
        print("All persons in the phone list:")
        print(phone_df.to_string(index=False))

while True:
    phone_list = []

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