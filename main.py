import numpy as np
from io import BytesIO
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import streamlit as st

st.title("Colour Clustering")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("Please upload an image to continue.")
else:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    save_path = f"uploaded_{uploaded_file.name}"
    # image.save(save_path)
    # st.success(f"Image saved as {save_path}")

    # Resize to reduce computation load
    image = image.resize((200, 200))

    img_np = np.array(image)

    pixels = img_np.reshape(-1, 3)

    # Compute WCSS for different numbers of clusters
    wcss = []
    for c in range(2, 11):
        kmeans = KMeans(n_clusters=c, random_state=0)
        kmeans.fit(pixels)
        wcss.append(kmeans.inertia_)

    # Plot WCSS (elbow method)
    st.subheader("WCSS (Within-Cluster Sum of Squares) vs. Number of Clusters")
    fig, ax = plt.subplots()
    ax.plot(range(2, 11), wcss, marker="D")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("WCSS")
    ax.set_title("Elbow Method for Optimal Clusters")
    ax.grid(True)
    st.pyplot(fig)

    n_colors = 7
    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(pixels)
    cluster_centers = kmeans.cluster_centers_.astype("uint8")
    labels = kmeans.labels_

    segmented_img = cluster_centers[labels].reshape(img_np.shape)

    st.subheader(f"Clustering Result with {n_colors} Colors")
    fig2, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(image)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(segmented_img)
    axs[1].set_title("Clustered Image")
    axs[1].axis("off")

    for i, color in enumerate(cluster_centers):
        axs[2].bar(i, 1, color=color / 255.0)
    axs[2].set_title("Extracted Color Palette")
    axs[2].set_xticks([])
    axs[2].set_yticks([])

    plt.tight_layout()
    st.pyplot(fig2)
    from io import BytesIO

    segmented_pil = Image.fromarray(segmented_img)

    img_buffer = BytesIO()
    segmented_pil.save(img_buffer, format="PNG")
    img_buffer.seek(0)  # Rewind the buffer
    st.download_button(
        label="Download Clustered Image",
        data=img_buffer,
        file_name="clustered_image.png",
        mime="image/png",
        icon=":material/download:",
    )
