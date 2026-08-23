#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <cctype>

namespace fs = std::filesystem;


// ============================================================
// CONFIGURATION
// ============================================================

const int INPUT_WIDTH  = 640;
const int INPUT_HEIGHT = 640;

const float DEFAULT_CONFIDENCE = 0.25f;
const float DEFAULT_NMS        = 0.45f;


// ============================================================
// DETECTION STRUCTURE
// ============================================================

struct Detection
{
    std::string label;

    float confidence;

    int x1;
    int y1;
    int x2;
    int y2;
};


// ============================================================
// GLOBAL CLASS NAMES
//
// IMPORTANT:
// The order MUST match the trained YOLO model's class order.
//
// Check your data.yaml before submission.
// ============================================================

std::vector<std::string> CLASS_NAMES =
{
    "gloved_hand",
    "bare_hand"
};


// ============================================================
// LABEL CONVERSION
// ============================================================

std::string convertLabel(
    std::string className
)
{
    std::transform(
        className.begin(),
        className.end(),
        className.begin(),
        [](unsigned char c)
        {
            return std::tolower(c);
        }
    );

    // glove / gloved
    if (
        className.find("glove") != std::string::npos &&
        className.find("no") == std::string::npos
    )
    {
        return "gloved_hand";
    }

    // no glove
    if (
        className.find("glove") != std::string::npos &&
        className.find("no") != std::string::npos
    )
    {
        return "bare_hand";
    }

    // bare
    if (
        className.find("bare") != std::string::npos
    )
    {
        return "bare_hand";
    }

    if (
        className.find("gloved") != std::string::npos
    )
    {
        return "gloved_hand";
    }

    return "";
}


// ============================================================
// JSON ESCAPE
// ============================================================

std::string escapeJson(
    const std::string& text
)
{
    std::string result;

    for (char c : text)
    {
        switch (c)
        {
            case '"':
                result += "\\\"";
                break;

            case '\\':
                result += "\\\\";
                break;

            case '\n':
                result += "\\n";
                break;

            case '\r':
                result += "\\r";
                break;

            case '\t':
                result += "\\t";
                break;

            default:
                result += c;
        }
    }

    return result;
}


// ============================================================
// SAVE JSON
//
// Output:
//
// {
//   "filename": "image1.jpg",
//   "detections": [
//      {
//        "label": "gloved_hand",
//        "confidence": 0.92,
//        "bbox": [x1,y1,x2,y2]
//      }
//   ]
// }
// ============================================================

void saveJson(
    const std::string& filename,
    const std::vector<Detection>& detections,
    const std::string& logDirectory
)
{
    fs::create_directories(
        logDirectory
    );

    fs::path imagePath(
        filename
    );

    std::string jsonFilename =
        imagePath.stem().string()
        + ".json";

    fs::path jsonPath =
        fs::path(logDirectory)
        / jsonFilename;

    std::ofstream file(
        jsonPath
    );

    if (!file.is_open())
    {
        std::cerr
            << "ERROR: Could not create JSON file: "
            << jsonPath
            << std::endl;

        return;
    }

    file
        << "{\n"
        << "    \"filename\": \""
        << escapeJson(filename)
        << "\",\n"
        << "    \"detections\": [\n";

    for (
        size_t i = 0;
        i < detections.size();
        ++i
    )
    {
        const Detection& d =
            detections[i];

        file
            << "        {\n";

        file
            << "            \"label\": \""
            << escapeJson(d.label)
            << "\",\n";

        file
            << "            \"confidence\": "
            << std::fixed
            << std::setprecision(4)
            << d.confidence
            << ",\n";

        file
            << "            \"bbox\": ["
            << d.x1
            << ", "
            << d.y1
            << ", "
            << d.x2
            << ", "
            << d.y2
            << "]\n";

        file
            << "        }";

        if (
            i + 1 <
            detections.size()
        )
        {
            file << ",";
        }

        file << "\n";
    }

    file
        << "    ]\n"
        << "}\n";

    file.close();

    std::cout
        << "JSON: "
        << jsonPath
        << std::endl;
}


// ============================================================
// DRAW DETECTIONS
//
// Green = gloved_hand
// Red   = bare_hand
// ============================================================

void drawDetections(
    cv::Mat& image,
    const std::vector<Detection>& detections
)
{
    for (
        const Detection& detection :
        detections
    )
    {
        cv::Scalar color;

        if (
            detection.label ==
            "gloved_hand"
        )
        {
            color =
                cv::Scalar(
                    0,
                    255,
                    0
                );
        }
        else
        {
            color =
                cv::Scalar(
                    0,
                    0,
                    255
                );
        }

        // ----------------------------------------------------
        // Bounding box
        // ----------------------------------------------------

        cv::rectangle(
            image,

            cv::Point(
                detection.x1,
                detection.y1
            ),

            cv::Point(
                detection.x2,
                detection.y2
            ),

            color,
            2
        );

        // ----------------------------------------------------
        // Label
        // ----------------------------------------------------

        std::ostringstream text;

        text
            << detection.label
            << " "
            << std::fixed
            << std::setprecision(2)
            << detection.confidence;

        std::string label =
            text.str();

        int baseline = 0;

        cv::Size textSize =
            cv::getTextSize(
                label,
                cv::FONT_HERSHEY_SIMPLEX,
                0.6,
                2,
                &baseline
            );

        int textX =
            detection.x1;

        int textY =
            detection.y1 - 10;

        if (
            textY < textSize.height + 5
        )
        {
            textY =
                textSize.height + 5;
        }

        // ----------------------------------------------------
        // Background behind text
        // ----------------------------------------------------

        cv::rectangle(
            image,

            cv::Point(
                textX,
                textY - textSize.height - 5
            ),

            cv::Point(
                textX + textSize.width + 5,
                textY
            ),

            color,
            cv::FILLED
        );

        // ----------------------------------------------------
        // Text
        // ----------------------------------------------------

        cv::putText(
            image,
            label,

            cv::Point(
                textX + 2,
                textY - 3
            ),

            cv::FONT_HERSHEY_SIMPLEX,
            0.6,

            cv::Scalar(
                255,
                255,
                255
            ),

            2,
            cv::LINE_AA
        );
    }
}


// ============================================================
// YOLO DETECTION
// ============================================================

std::vector<Detection> detect(
    cv::dnn::Net& net,
    const cv::Mat& image,
    float confidenceThreshold,
    float nmsThreshold
)
{
    std::vector<Detection> detections;

    if (image.empty())
    {
        return detections;
    }

    // ========================================================
    // IMAGE PREPROCESSING
    // ========================================================

    cv::Mat blob;

    cv::dnn::blobFromImage(
        image,

        blob,

        1.0 / 255.0,

        cv::Size(
            INPUT_WIDTH,
            INPUT_HEIGHT
        ),

        cv::Scalar(),

        true,

        false
    );

    net.setInput(
        blob
    );

    // ========================================================
    // FORWARD PASS
    // ========================================================

    std::vector<cv::Mat> outputs;

    net.forward(
        outputs,
        net.getUnconnectedOutLayersNames()
    );

    if (
        outputs.empty()
    )
    {
        std::cerr
            << "ERROR: Model returned no output."
            << std::endl;

        return detections;
    }

    cv::Mat output =
        outputs[0];

    // ========================================================
    // YOLO OUTPUT
    //
    // Typical YOLO11 ONNX:
    //
    // [1, 4 + num_classes, predictions]
    //
    // Example:
    //
    // [1, 6, 8400]
    //
    // Convert to:
    //
    // [8400, 6]
    // ========================================================

    int rows = 0;

    int dimensions = 0;

    if (
        output.dims == 3
    )
    {
        dimensions =
            output.size[1];

        rows =
            output.size[2];

        output =
            cv::Mat(
                dimensions,
                rows,
                CV_32F,
                output.ptr<float>()
            ).t();
    }
    else if (
        output.dims == 2
    )
    {
        rows =
            output.rows;

        dimensions =
            output.cols;
    }
    else
    {
        std::cerr
            << "ERROR: Unsupported output dimensions: "
            << output.dims
            << std::endl;

        return detections;
    }

    // ========================================================
    // SCALE FACTORS
    // ========================================================

    float xFactor =
        static_cast<float>(
            image.cols
        )
        /
        static_cast<float>(
            INPUT_WIDTH
        );

    float yFactor =
        static_cast<float>(
            image.rows
        )
        /
        static_cast<float>(
            INPUT_HEIGHT
        );

    // ========================================================
    // TEMPORARY DATA
    // ========================================================

    std::vector<cv::Rect> boxes;

    std::vector<float> confidences;

    std::vector<int> classIds;

    // ========================================================
    // PROCESS EVERY YOLO PREDICTION
    // ========================================================

    for (
        int i = 0;
        i < rows;
        ++i
    )
    {
        // ----------------------------------------------------
        // YOLO coordinates
        // ----------------------------------------------------

        float centerX =
            output.at<float>(
                i,
                0
            );

        float centerY =
            output.at<float>(
                i,
                1
            );

        float width =
            output.at<float>(
                i,
                2
            );

        float height =
            output.at<float>(
                i,
                3
            );

        // ----------------------------------------------------
        // Class scores
        // ----------------------------------------------------

        cv::Mat classScores =
            output.row(i).colRange(
                4,
                dimensions
            );

        cv::Point classIdPoint;

        double maxClassScore;

        cv::minMaxLoc(
            classScores,

            nullptr,

            &maxClassScore,

            nullptr,

            &classIdPoint
        );

        float confidence =
            static_cast<float>(
                maxClassScore
            );

        // ----------------------------------------------------
        // Confidence filtering
        // ----------------------------------------------------

        if (
            confidence <
            confidenceThreshold
        )
        {
            continue;
        }

        int classId =
            classIdPoint.x;

        // ----------------------------------------------------
        // Check class
        // ----------------------------------------------------

        if (
            classId < 0 ||
            classId >=
            static_cast<int>(
                CLASS_NAMES.size()
            )
        )
        {
            continue;
        }

        // ----------------------------------------------------
        // Convert YOLO coordinates
        // ----------------------------------------------------

        int left =
            static_cast<int>(
                (
                    centerX
                    - width / 2.0f
                )
                *
                xFactor
            );

        int top =
            static_cast<int>(
                (
                    centerY
                    - height / 2.0f
                )
                *
                yFactor
            );

        int boxWidth =
            static_cast<int>(
                width
                *
                xFactor
            );

        int boxHeight =
            static_cast<int>(
                height
                *
                yFactor
            );

        // ----------------------------------------------------
        // Create rectangle
        // ----------------------------------------------------

        cv::Rect box(
            left,
            top,
            boxWidth,
            boxHeight
        );

        // ----------------------------------------------------
        // Keep box inside image
        // ----------------------------------------------------

        box &=
            cv::Rect(
                0,
                0,
                image.cols,
                image.rows
            );

        if (
            box.width <= 0 ||
            box.height <= 0
        )
        {
            continue;
        }

        boxes.push_back(
            box
        );

        confidences.push_back(
            confidence
        );

        classIds.push_back(
            classId
        );
    }

    // ========================================================
    // NON-MAXIMUM SUPPRESSION
    // ========================================================

    std::vector<int> indices;

    cv::dnn::NMSBoxes(
        boxes,
        confidences,
        confidenceThreshold,
        nmsThreshold,
        indices
    );

    // ========================================================
    // FINAL DETECTIONS
    // ========================================================

    for (
        int index :
        indices
    )
    {
        if (
            index < 0 ||
            index >=
            static_cast<int>(
                boxes.size()
            )
        )
        {
            continue;
        }

        Detection detection;

        detection.label =
            convertLabel(
                CLASS_NAMES[
                    classIds[index]
                ]
            );

        // ----------------------------------------------------
        // Ignore unknown class
        // ----------------------------------------------------

        if (
            detection.label.empty()
        )
        {
            continue;
        }

        detection.confidence =
            confidences[index];

        detection.x1 =
            boxes[index].x;

        detection.y1 =
            boxes[index].y;

        detection.x2 =
            boxes[index].x
            +
            boxes[index].width;

        detection.y2 =
            boxes[index].y
            +
            boxes[index].height;

        detections.push_back(
            detection
        );
    }

    return detections;
}


// ============================================================
// PRINT HELP
// ============================================================

void printHelp()
{
    std::cout
        << "\nUsage:\n\n"

        << "glove_detector "
        << "--input input "
        << "--output output "
        << "--confidence 0.25 "
        << "--model models/glove_detector.onnx\n\n"

        << "Arguments:\n"

        << "  --input       Input image folder\n"

        << "  --output      Output image folder\n"

        << "  --confidence  Confidence threshold\n"

        << "  --model       ONNX model path\n\n";
}


// ============================================================
// MAIN
// ============================================================

int main(
    int argc,
    char* argv[]
)
{
    // ========================================================
    // DEFAULT VALUES
    // ========================================================

    std::string modelPath =
        "models/glove_detector.onnx";

    std::string inputFolder =
        "input";

    std::string outputFolder =
        "output";

    std::string logFolder =
        "logs";

    float confidenceThreshold =
        DEFAULT_CONFIDENCE;

    float nmsThreshold =
        DEFAULT_NMS;

    // ========================================================
    // COMMAND LINE ARGUMENTS
    // ========================================================

    for (
        int i = 1;
        i < argc;
        ++i
    )
    {
        std::string arg =
            argv[i];

        if (
            arg == "--help" ||
            arg == "-h"
        )
        {
            printHelp();

            return 0;
        }

        else if (
            arg == "--input"
            &&
            i + 1 < argc
        )
        {
            inputFolder =
                argv[++i];
        }

        else if (
            arg == "--output"
            &&
            i + 1 < argc
        )
        {
            outputFolder =
                argv[++i];
        }

        else if (
            arg == "--confidence"
            &&
            i + 1 < argc
        )
        {
            confidenceThreshold =
                std::stof(
                    argv[++i]
                );
        }

        else if (
            arg == "--model"
            &&
            i + 1 < argc
        )
        {
            modelPath =
                argv[++i];
        }

        else
        {
            std::cerr
                << "Unknown argument: "
                << arg
                << std::endl;

            printHelp();

            return -1;
        }
    }

    // ========================================================
    // HEADER
    // ========================================================

    std::cout
        << "\n"
        << "============================================\n"
        << "       GLOVE DETECTION SYSTEM - C++\n"
        << "============================================\n"
        << "\n";

    std::cout
        << "Model       : "
        << modelPath
        << "\n";

    std::cout
        << "Input       : "
        << inputFolder
        << "\n";

    std::cout
        << "Output      : "
        << outputFolder
        << "\n";

    std::cout
        << "Logs        : "
        << logFolder
        << "\n";

    std::cout
        << "Confidence  : "
        << confidenceThreshold
        << "\n";

    std::cout
        << "NMS         : "
        << nmsThreshold
        << "\n";

    // ========================================================
    // CREATE DIRECTORIES
    // ========================================================

    try
    {
        fs::create_directories(
            inputFolder
        );

        fs::create_directories(
            outputFolder
        );

        fs::create_directories(
            logFolder
        );
    }
    catch (
        const std::exception& e
    )
    {
        std::cerr
            << "ERROR creating directories: "
            << e.what()
            << std::endl;

        return -1;
    }

    // ========================================================
    // CHECK MODEL
    // ========================================================

    if (
        !fs::exists(
            modelPath
        )
    )
    {
        std::cerr
            << "\nERROR: ONNX model not found:\n"
            << modelPath
            << "\n\n";

        std::cerr
            << "Put your model here:\n"
            << "models/glove_detector.onnx\n";

        return -1;
    }

    // ========================================================
    // LOAD MODEL
    // ========================================================

    cv::dnn::Net net;

    try
    {
        std::cout
            << "\nLoading YOLO ONNX model..."
            << std::endl;

        net =
            cv::dnn::readNetFromONNX(
                modelPath
            );

        if (
            net.empty()
        )
        {
            std::cerr
                << "ERROR: Model is empty."
                << std::endl;

            return -1;
        }

        net.setPreferableBackend(
            cv::dnn::DNN_BACKEND_OPENCV
        );

        net.setPreferableTarget(
            cv::dnn::DNN_TARGET_CPU
        );
    }
    catch (
        const cv::Exception& e
    )
    {
        std::cerr
            << "\nERROR loading ONNX model:\n"
            << e.what()
            << std::endl;

        return -1;
    }

    std::cout
        << "Model loaded successfully.\n";

    // ========================================================
    // DISPLAY CLASSES
    // ========================================================

    std::cout
        << "\nClasses:\n";

    for (
        size_t i = 0;
        i < CLASS_NAMES.size();
        ++i
    )
    {
        std::cout
            << "  "
            << i
            << " -> "
            << CLASS_NAMES[i]
            << "\n";
    }

    // ========================================================
    // FIND INPUT IMAGES
    // ========================================================

    std::vector<fs::path>
        imageFiles;

    try
    {
        for (
            const auto& entry :
            fs::directory_iterator(
                inputFolder
            )
        )
        {
            if (
                !entry.is_regular_file()
            )
            {
                continue;
            }

            std::string extension =
                entry.path()
                    .extension()
                    .string();

            std::transform(
                extension.begin(),
                extension.end(),
                extension.begin(),
                [](unsigned char c)
                {
                    return std::tolower(c);
                }
            );

            if (
                extension == ".jpg" ||
                extension == ".jpeg" ||
                extension == ".png" ||
                extension == ".bmp" ||
                extension == ".webp"
            )
            {
                imageFiles.push_back(
                    entry.path()
                );
            }
        }
    }
    catch (
        const std::exception& e
    )
    {
        std::cerr
            << "ERROR reading input folder: "
            << e.what()
            << std::endl;

        return -1;
    }

    // ========================================================
    // SORT FILES
    // ========================================================

    std::sort(
        imageFiles.begin(),
        imageFiles.end()
    );

    std::cout
        << "\nFound "
        << imageFiles.size()
        << " images.\n";

    // ========================================================
    // NO IMAGE
    // ========================================================

    if (
        imageFiles.empty()
    )
    {
        std::cout
            << "\nNo images found.\n"
            << "Put .jpg images inside:\n"
            << inputFolder
            << "\n";

        return 0;
    }

    // ========================================================
    // PROCESS ALL IMAGES
    // ========================================================

    int processed =
        0;

    int failed =
        0;

    int totalDetections =
        0;

    int glovedCount =
        0;

    int bareCount =
        0;

    for (
        size_t i = 0;
        i < imageFiles.size();
        ++i
    )
    {
        fs::path imagePath =
            imageFiles[i];

        std::string filename =
            imagePath.filename()
                .string();

        std::cout
            << "\n--------------------------------------------\n"
            << "Image "
            << (i + 1)
            << " / "
            << imageFiles.size()
            << ": "
            << filename
            << "\n"
            << "--------------------------------------------\n";

        // ====================================================
        // READ IMAGE
        // ====================================================

        cv::Mat image =
            cv::imread(
                imagePath.string()
            );

        if (
            image.empty()
        )
        {
            std::cerr
                << "ERROR: Could not read image."
                << std::endl;

            failed++;

            continue;
        }

        // ====================================================
        // DETECTION
        // ====================================================

        std::vector<Detection>
            detections;

        try
        {
            detections =
                detect(
                    net,
                    image,
                    confidenceThreshold,
                    nmsThreshold
                );
        }
        catch (
            const cv::Exception& e
        )
        {
            std::cerr
                << "ERROR during inference:\n"
                << e.what()
                << std::endl;

            failed++;

            continue;
        }

        // ====================================================
        // DRAW
        // ====================================================

        drawDetections(
            image,
            detections
        );

        // ====================================================
        // SAVE ANNOTATED IMAGE
        // ====================================================

        fs::path outputPath =
            fs::path(
                outputFolder
            )
            /
            filename;

        bool saved =
            cv::imwrite(
                outputPath.string(),
                image
            );

        if (
            !saved
        )
        {
            std::cerr
                << "WARNING: Could not save output image."
                << std::endl;
        }
        else
        {
            std::cout
                << "Annotated image saved: "
                << outputPath
                << "\n";
        }

        // ====================================================
        // SAVE JSON
        // ====================================================

        saveJson(
            filename,
            detections,
            logFolder
        );

        // ====================================================
        // PRINT DETECTIONS
        // ====================================================

        if (
            detections.empty()
        )
        {
            std::cout
                << "No detections.\n";
        }

        for (
            const Detection& d :
            detections
        )
        {
            std::cout
                << "Label      : "
                << d.label
                << "\n";

            std::cout
                << "Confidence : "
                << std::fixed
                << std::setprecision(4)
                << d.confidence
                << "\n";

            std::cout
                << "BBox       : ["
                << d.x1
                << ", "
                << d.y1
                << ", "
                << d.x2
                << ", "
                << d.y2
                << "]\n";

            totalDetections++;

            if (
                d.label ==
                "gloved_hand"
            )
            {
                glovedCount++;
            }
            else if (
                d.label ==
                "bare_hand"
            )
            {
                bareCount++;
            }
        }

        processed++;
    }

    // ========================================================
    // FINAL SUMMARY
    // ========================================================

    std::cout
        << "\n\n"
        << "============================================\n"
        << "              FINAL SUMMARY\n"
        << "============================================\n";

    std::cout
        << "Images found       : "
        << imageFiles.size()
        << "\n";

    std::cout
        << "Images processed   : "
        << processed
        << "\n";

    std::cout
        << "Images failed      : "
        << failed
        << "\n";

    std::cout
        << "Total detections   : "
        << totalDetections
        << "\n";

    std::cout
        << "Gloved hands       : "
        << glovedCount
        << "\n";

    std::cout
        << "Bare hands         : "
        << bareCount
        << "\n";

    std::cout
        << "Output folder      : "
        << outputFolder
        << "\n";

    std::cout
        << "Logs folder        : "
        << logFolder
        << "\n";

    std::cout
        << "============================================\n";

    return 0;
}