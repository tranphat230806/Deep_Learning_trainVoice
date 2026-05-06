// JavaScript for Bedroom Page
function toggleDoor() {
  const doorImage = document.getElementById("door-image");
  const isOpen = doorImage.dataset.state === "open";

  if (isOpen) {
    doorImage.src = "https://via.placeholder.com/600x400?text=Door+Closed";
    doorImage.dataset.state = "closed";
  } else {
    doorImage.src = "https://via.placeholder.com/600x400?text=Door+Open";
    doorImage.dataset.state = "open";
  }
}
