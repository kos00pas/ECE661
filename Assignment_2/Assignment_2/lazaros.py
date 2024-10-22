from pymodbus.client import ModbusTcpClient
import socket
import time

# Get IP Address
hostname = socket.gethostname()
IPAddr = socket.gethostbyname(hostname)


# Connect with OpenPLC runtime
client = ModbusTcpClient(IPAddr)
isConnected = client.connect()
print(hostname)
print(IPAddr)
print(client)
# print(isConnected)

def continues_disruption():
    try:
        if client.connect():
            print("Start continues disruption")
            client.write_coil(6, 0, 1)   # middle splitter
            client.write_coil(5, 0, 1)   # right conveyor
            client.write_coil(3, 0, 1)   # left conveyor
            client.write_coil(7, 0, 1)   # middle/forward conveyor
            client.write_coil(6, True)  # Turn on middle splitter
        else:
            print("Problem: Not connected")
    except Exception as e:
        print(f"An error occurred: {e}")

def read_all_coils(start_address=0, count=10):
        print("Reading coils...")
        for address in range(start_address, start_address + count):
            response = client.read_coils(address, 1)
            if not response.isError():
                print(f"Coil at address {address}: {response.bits[0]}")
            else:
                print(f"Error reading coil at address {address}")

    # Define function to read all holding registers in a range
def read_all_holding_registers(start_address=0, count=10):
        print("Reading holding registers...")
        for address in range(start_address, start_address + count):
            response = client.read_holding_registers(address, 1)
            if not response.isError():
                print(f"Holding Register at address {address}: {response.registers[0]}")
            else:
                print(f"Error reading holding register at address {address}")

    # Scan coils and holding registers in specific address ranges
    # Adjust start_address and count as necessary
read_all_coils(start_address=0, count=100)  # Read first 100 coils
read_all_holding_registers(start_address=0, count=100)  # Read first 100 holding registers

    # time.sleep(0.1)  # Small delay for PLC response
    #
    # middle_splitter = client.read_coils(6, 1)
    # if not middle_splitter.isError():
    #     print("Middle Splitter State:", middle_splitter.bits[0])
    # else:
    #     print("Error reading Middle Splitter:", middle_splitter)

#     # Read the discrete inputs
#     start = client.read_discrete_inputs(100 * 16 + 9, 1)  # %IX100.9
#     scale_sensor = client.read_discrete_inputs(100 * 16 + 1, 1)  # %IX100.1
#
#     # Read the coils
#     main_conveyor = client.read_coils(100 * 16 + 0, 1)  # %QX100.0
#     right_conveyor = client.read_coils(100 * 16 + 5, 1)  # %QX100.5
#     left_conveyor = client.read_coils(100 * 16 + 3, 1)  # %QX100.3
#     middle_conveyor = client.read_coils(100 * 16 + 7, 1)  # %QX100.7
#     scale_conveyor = client.read_coils(100 * 16 + 1, 1)  # %QX100.1
#     send_forward = client.read_coils(100 * 16 + 6, 1)  # %QX100.6
#     send_left = client.read_coils(100 * 16 + 2, 1)  # %QX100.2
#     send_right = client.read_coils(100 * 16 + 4, 1)  # %QX100.4
#
#     # Read the input register for the weight
#     weight = client.read_input_registers(100, 1)  # %IW100
#
#     # Display the values
#     print("Start:", start.bits[0] if start.isError() == False else "Error")
#     print("Scale Sensor:", scale_sensor.bits[0] if scale_sensor.isError() == False else "Error")
#     print("Main Conveyor:", main_conveyor.bits[0] if main_conveyor.isError() == False else "Error")
#     print("Right Conveyor:", right_conveyor.bits[0] if right_conveyor.isError() == False else "Error")
#     print("Left Conveyor:", left_conveyor.bits[0] if left_conveyor.isError() == False else "Error")
#     print("Middle Conveyor:", middle_conveyor.bits[0] if middle_conveyor.isError() == False else "Error")
#     print("Scale Conveyor:", scale_conveyor.bits[0] if scale_conveyor.isError() == False else "Error")
#     print("Send Forward:", send_forward.bits[0] if send_forward.isError() == False else "Error")
#     print("Send Left:", send_left.bits[0] if send_left.isError() == False else "Error")
#     print("Send Right:", send_right.bits[0] if send_right.isError() == False else "Error")
#     print("Weight:", weight.registers[0] if weight.isError() == False else "Error")
#     time.sleep(3)

print("End")
# Close connection
client.close()