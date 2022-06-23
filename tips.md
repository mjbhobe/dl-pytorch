## Some tips I gathered from the Internet on Pytorch, Colab, Deep Learning

1. 

3. How to prevent Colab Notebook from disconnecting runtime during long running job & other tips: [read this article on Medium.com](https://towardsdatascience.com/10-tips-for-a-better-google-colab-experience-33f8fe721b82#8c1e)<br/>Open your Chrome DevTools by pressing F12 or ctrl+shift+i on Linux and enter the following JavaScript snippet in your console:
```javascript
function KeepClicking() {
    console.log("Clicking");
    document.querySelector("colab-connect-button").click()
}
setInterval(KeepClicking,60000)
```