import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image


def main():
    st.title("🌈🌈 K Dominant Colors 🌈🌈")
    st.header("Upload an Image 😎😎")
    # Image
    image_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'])
    if image_file is not None:
        st.header("Original Image")
        img = Image.open(image_file)
        img = img.resize((500,500))
        st.image(img)

        # Number of clusters
        st.header("Select a suitable number of clusters ")
        clusters = st.slider("K" , 1, 20 , key="clusters")

        if st.button("Go",key="Go"):
            kDominantColors(img,clusters)


    if st.checkbox("About",key="About"):
        about()


def kDominantColors(img,k):

    img_array = np.array(img)
    img_pixels = img_array.reshape((img_array.shape[0]*img_array.shape[1],3))
    km = KMeans(n_clusters=k)
    km.fit(img_pixels)

    # Color Centers
    colors = np.array(km.cluster_centers_,dtype='uint8')

    # new image
    new_img = np.zeros((img_array.shape[0]*img_array.shape[1], 3),dtype='uint8')

    # Filling the dominant colors according to the fitted/assigned labels by KMeans Algo
    for ix in range(new_img.shape[0]):
        new_img[ix] = colors[km.labels_[ix]]

    # reshaping the image
    new_img = new_img.reshape(img_array.shape)

    st.header("Resultant Image with k = {}".format(k))
    st.image(new_img)


def about():


	st.write(
		"""# K Dominant Colors
        It uses K Means Algorithm to select K most dominant color of the image.
        After that, this program will display the image uploaded within those K colors itself.
        """)


if __name__ == '__main__':
    main()
