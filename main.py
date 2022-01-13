####################################################
#                                                  #
# Ethan Boulanger                                  #
# CS 447                                           #
# Implementation for Graduate Standing Project     #
#                                                  #
####################################################


import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp
import math
from random import randrange


def main():
    #Input key to the chaotic algorithm
    #x0, a, b
    key = (0.3, 1.57, 3.5)

    # Read in the image
    temp = plt.imread("sample.jpg")
    plt.imshow(temp, cmap='gray')
    plt.title("Original Image")
    plt.show()

    # Make a copy of the image so that we can modify it
    im_array = np.copy(temp)

    # Remove third dimension from image (since it's greyscale)
    originalImage = im_array[:, :, 0]

    # Encrypt the image
    encryptedImage = encrypt(originalImage, key)

    # Save the new image
    plt.imsave("encrypted.jpg", encryptedImage, cmap='gray')

    # Decrypt the image
    decryptedImage = decrypt(encryptedImage, key)

    # Save the new image
    plt.imsave("decrypted.jpg", decryptedImage, cmap='gray')

    # Let's take a look at the input data vs. encrypted date
    histogram(originalImage, "Histogram of the original image")
    histogram(encryptedImage, "Histogram of the encrypted image")
    correlationPlot(originalImage, 500, "Correlation Plot of original image")
    correlationPlot(encryptedImage, 500, "Correlation Plot of encrypted image")


def encrypt(im_array, key):
    # Flatten 2D array into 1D for linear processing
    im_array = im_array.flatten()
    length = len(im_array)

    x = float(key[0])
    y = float(0)
    alpha = key[1]
    beta = key[2]

    # Run chaotic sequence 100 times before using data
    for i in range(0,100):
        x = float((1 - beta ** -4) * mp.cot((alpha) / (1 + beta)) * ((1 + (1 / beta)) ** beta) * mp.tan(alpha * x) * ((1 - x) ** beta))

    # Encrypt the image, 5 pixels (bytes) at a time
    for i in range(0, length, 5):
        # Run through chaotic sequence 3 times, only caring about third iteration
        for j in range(0, 3):
            x = float((1 - beta ** -4) * mp.cot((alpha) / (1 + beta)) * ((1 + (1 / beta)) ** beta) * mp.tan(alpha * x) * ((1 - x) ** beta))
            # Save value to 15 decimal places
            y = f"{x:.15f}"

        # Convert the float to a string for easy splitting
        string = str(y)

        # Split the string into 5 chunks of 3 digits, use mod 256 to put it in correct range of 0-255
        num1 = int(string[2:5]) % 256
        num2 = int(string[5:8]) % 256
        num3 = int(string[8:11]) % 256
        num4 = int(string[11:14]) % 256
        num5 = int(string[14:17]) % 256

        # XOR the number with a pixel of an image
        im_array[i] = im_array[i] ^ num1
        im_array[i+1] = im_array[i+1] ^ num2
        im_array[i+2] = im_array[i+2] ^ num3
        im_array[i+3] = im_array[i+3] ^ num4
        im_array[i+4] = im_array[i+4] ^ num5

    # Reconstruct 2D array from flattened 1D array
    im_array = np.reshape(im_array, (225, 225))

    # Show the new image
    plt.imshow(im_array, cmap='gray')
    plt.title("Encrypted Image")
    plt.show()

    return im_array


def decrypt(im_array, key):
    # This is pretty much the exact same as decrypt because stream ciphers work in reverse
    # Flatten 2D array into 1D for linear processing
    im_array = im_array.flatten()
    length = len(im_array)

    x = float(key[0])
    y = float(0)
    alpha = key[1]
    beta = key[2]

    # Run chaotic sequence 100 times before using data
    for i in range(0,100):
        x = float((1 - beta ** -4) * mp.cot((alpha) / (1 + beta)) * ((1 + (1 / beta)) ** beta) * mp.tan(alpha * x) * ((1 - x) ** beta))

    # Encrypt the image, 5 pixels (bytes) at a time
    for i in range(0, length, 5):
        # Run through chaotic sequence 3 times, only caring about third iteration
        for j in range(0, 3):
            x = float((1 - beta ** -4) * mp.cot((alpha) / (1 + beta)) * ((1 + (1 / beta)) ** beta) * mp.tan(alpha * x) * ((1 - x) ** beta))
            # Save value to 15 decimal places
            y = f"{x:.15f}"

        # Convert the float to a string for easy splitting
        string = str(y)

        # Split the string into 5 chunks of 3 digits, use mod 256 to put it in correct range of 0-255
        num1 = int(string[2:5]) % 256
        num2 = int(string[5:8]) % 256
        num3 = int(string[8:11]) % 256
        num4 = int(string[11:14]) % 256
        num5 = int(string[14:17]) % 256

        # XOR the number with a pixel of an image
        im_array[i] = im_array[i] ^ num1
        im_array[i+1] = im_array[i+1] ^ num2
        im_array[i+2] = im_array[i+2] ^ num3
        im_array[i+3] = im_array[i+3] ^ num4
        im_array[i+4] = im_array[i+4] ^ num5

    # Reconstruct 2D array from flattened 1D array
    im_array = np.reshape(im_array, (225, 225))

    # Show the new image
    plt.imshow(im_array, cmap='gray')
    plt.title("Decrypted Image")
    plt.show()

    return im_array


def histogram(im_array, title):
    # Flatten into 1D array for linear processing
    im_array = im_array.flatten()

    # Make a histogram with 256 bins
    plt.hist(im_array, 256)

    # Show the histogram
    plt.xlim([0, 255])
    plt.ylim([0,600])
    plt.title(title)
    plt.xlabel("Intensity of Random Pixel")
    plt.ylabel("Occurrences of Pixel Intensity")
    plt.show()


def correlationPlot(im_array, sampleSize, title):
    # Flatten 2D array into 1D for linear processing
    im_array = im_array.flatten()
    length = len(im_array)

    # Make a x and y list
    xPoints = list()
    yPoints = list()

    # Choose N samples and get the adjacent pixel as well
    for i in range(0, sampleSize):
        index = randrange(0, length - 2)
        xPoints.append(im_array[index])
        yPoints.append(im_array[index + 1])

    # Determine the correlation
    # Find the average x and y
    xAvg = 0
    yAvg = 0
    for i in range(0, sampleSize):
        xAvg += xPoints[i]
        yAvg += yPoints[i]
    xAvg = xAvg / sampleSize
    yAvg = yAvg / sampleSize

    # Find the covariance
    covariance = 0
    for i in range(0, sampleSize):
        covariance += (xPoints[i] - xAvg)*(yPoints[i] - yAvg)
    covariance = covariance / sampleSize

    # Find the standard deviation of x and y
    stdX = 0
    stdY = 0
    for i in range(0, sampleSize):
        stdX += (xPoints[i] - xAvg)*(xPoints[i] - xAvg)
        stdY += (yPoints[i] - yAvg)*(yPoints[i] - yAvg)
    stdX = stdX / sampleSize
    stdY = stdY / sampleSize

    # Put it all together
    correlation = covariance/((math.sqrt(stdX))*(math.sqrt(stdY)))
    correlation = f"{correlation:.5f}"

    # Plot the points
    plt.scatter(xPoints, yPoints)
    plt.title(title)
    plt.xlabel("Intensity of Random Pixel")
    plt.ylabel("Intensity of Random Pixel + 1")
    plt.xlim([0, 255])
    plt.ylim([0, 255])
    plt.text(10, 220, "Correlation: " + str(correlation), fontsize=10,  bbox={'facecolor': 'red', 'alpha': 0.8, 'pad': 4})
    plt.show()


if __name__ == "__main__":
    main()
